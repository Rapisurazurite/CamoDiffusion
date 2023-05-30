import torch
from einops import rearrange, pack, unpack
from torch import nn

from denoising_diffusion_pytorch.simple_diffusion import *


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn

        class LayerNorm(nn.Module):
            def __init__(self, dim):
                super().__init__()
                self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

            def forward(self, x):
                eps = 1e-5 if x.dtype == torch.float32 else 1e-3
                var = torch.var(x, dim=1, unbiased=False, keepdim=True)
                mean = torch.mean(x, dim=1, keepdim=True)
                return (x - mean) * (var + eps).rsqrt() * self.g

        self.norm = LayerNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


class CondUnetWrapper(nn.Module):
    def __init__(self, unet, feature_exactor, translayer=None):
        super().__init__()
        self.feature_exactor = feature_exactor
        self.unet = unet
        self.translayer = translayer


    def extract_features(self, cond_img):
        features = self.feature_exactor(cond_img)
        features = self.translayer(features) if self.translayer is not None else features
        return features

    def sample_unet(self, x, times, conditioning_features):
        return self.unet(x, times, conditioning_features)

    def forward(self, x, times, cond_img):
        conditioning_features = self.extract_features(cond_img)
        return self.sample_unet(x, times, conditioning_features)


class CondUViT(UViT):
    def __init__(self, dim, init_dim=None, out_dim=None, dim_mults=(1, 2, 4, 8), downsample_factor=2, channels=3,
                 out_channels=None, vit_depth=6, vit_dropout=0.2, attn_dim_head=32, attn_heads=4, ff_mult=4,
                 resnet_block_groups=8, learned_sinusoidal_dim=16, init_img_transform: callable = None,
                 final_img_itransform: callable = None, patch_size=1, dual_patchnorm=False, conditioning_klass=None,
                 use_condtionning=(True, True, True, True), condition_dims=None, condition_sizes=None,
                 skip_connect_condition_fmaps=False):
        super().__init__(dim, init_dim, out_dim, dim_mults, downsample_factor, channels, out_channels, vit_depth,
                         vit_dropout,
                         attn_dim_head, attn_heads, ff_mult, resnet_block_groups, learned_sinusoidal_dim,
                         init_img_transform, final_img_itransform, patch_size, dual_patchnorm)
        assert conditioning_klass is not None, \
            "Conditioning class must be provided, which is a class that can be instantiated with fmap_size and dim_in"
        assert len(use_condtionning) == len(condition_dims) == len(condition_sizes), \
            "Conditioning parameters must be of the same length"

        self.skip_connect_condition_fmaps = skip_connect_condition_fmaps
        # Calculate some parameters
        init_dim = default(init_dim, dim)
        dims = [init_dim, *map(lambda m: int(dim * m), dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        resnet_block = partial(ResnetBlock, groups=resnet_block_groups)

        time_dim = dim * 4

        downsample_factor = cast_tuple(downsample_factor, len(dim_mults))
        num_resolutions = len(in_out)
        assert num_resolutions + 1 == len(use_condtionning), \
            "Condition parameter have an extra original size feature map"

        # Add conditioners
        self.conditioners = nn.ModuleList([])
        for ind, (use_cond, cond_dim, cond_size) in enumerate(zip(use_condtionning, condition_dims, condition_sizes)):
            if use_cond:
                self.conditioners.append(conditioning_klass(cond_size, cond_dim))
            else:
                self.conditioners.append(None)

        # Rewrite downsampling and upsampling
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])

        for ind, ((dim_in, dim_out), factor) in enumerate(zip(in_out, downsample_factor)):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                resnet_block(dim_in, dim_in, time_emb_dim=time_dim),
                resnet_block(dim_in, dim_in, time_emb_dim=time_dim),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                Downsample(dim_in, dim_out, factor=factor)
            ]))

        for ind, ((dim_in, dim_out), factor, cond_dim) \
                in enumerate(zip(reversed(in_out), reversed(downsample_factor),
                                 reversed(condition_dims[:-1]))):
            is_last = ind == (len(in_out) - 1)
            skip_connect_dim = cond_dim if self.skip_connect_condition_fmaps else 0

            self.ups.append(nn.ModuleList([
                Upsample(dim_out, dim_in, factor=factor),
                resnet_block(dim_in * 2 + skip_connect_dim, dim_in, time_emb_dim=time_dim),
                resnet_block(dim_in * 2 + skip_connect_dim, dim_in, time_emb_dim=time_dim),
                LinearAttention(dim_in),
            ]))

    def forward(self, x, times, cond):
        skip_connect_c = self.skip_connect_condition_fmaps
        assert len(cond) == len(self.conditioners)

        x = self.init_img_transform(x)
        x = self.init_conv(x)  # (B, X, H, W) -> (B, X, H/patch_size, W/patch_size)
        r = x.clone()

        t = self.time_mlp(times)

        h = []

        for (block1, block2, attn, downsample), cond_feature, conditioner in zip(self.downs, cond, self.conditioners):
            x = block1(x, t)
            h.append([x, cond_feature] if skip_connect_c else [x])

            x = block2(x, t)
            x = attn(x)

            # DO CONDITIONING
            x = (x + conditioner(x, cond_feature)) if conditioner is not None else x

            h.append([x, cond_feature] if skip_connect_c else [x])
            x = downsample(x)

        x = (x + self.conditioners[-1](x, cond[-1])) if self.conditioners[-1] is not None else x

        x = rearrange(x, 'b c h w -> b h w c')
        x, ps = pack([x], 'b * c')

        x = self.vit(x, t)

        x, = unpack(x, ps, 'b * c')
        x = rearrange(x, 'b h w c -> b c h w')

        for upsample, block1, block2, attn in self.ups:
            x = upsample(x)

            x = torch.cat((x, *h.pop()), dim=1)
            x = block1(x, t)

            x = torch.cat((x, *h.pop()), dim=1)
            x = block2(x, t)
            x = attn(x)

        x = torch.cat((x, r), dim=1)

        x = self.final_res_block(x, t)
        x = self.final_conv(x)

        x = self.unpatchify(x)
        return self.final_img_itransform(x)


class CondGaussianDiffusion(GaussianDiffusion):
    def __init__(
            self,
            model: UViT,
            *,
            image_size,
            channels=1,
            extra_channels=0,
            cond_channels=3,
            pred_objective='v',
            loss_type='l2',
            noise_schedule=logsnr_schedule_cosine,
            noise_d=None,
            noise_d_low=None,
            noise_d_high=None,
            num_sample_steps=500,
            clip_sample_denoised=True,
    ):
        super(CondGaussianDiffusion, self).__init__(model, image_size=image_size, channels=channels,
                                                    pred_objective=pred_objective, noise_schedule=noise_schedule,
                                                    noise_d=noise_d, noise_d_low=noise_d_low, noise_d_high=noise_d_high,
                                                    num_sample_steps=num_sample_steps,
                                                    clip_sample_denoised=clip_sample_denoised)
        if loss_type not in ['l2', 'l1', 'l1+l2', 'mean(l1, l2)']:
            try:
                from utils.import_utils import get_obj_from_str
                loss_type = get_obj_from_str(loss_type)
            except:
                raise NotImplementedError

        self.loss_type = loss_type
        self.extra_channels = extra_channels
        self.cond_channels = cond_channels

        # This history is used to store the history of the x_start in reverse diffusion process
        # When call the sample function, the history will be reset and store the x_start of each sample
        self.history = []

    def forward(self, img, cond_img, seg=None, extra_cond=None, *args, **kwargs):
        b, channels, h, w = img.shape
        cond_channels = cond_img.shape[1]
        assert channels == self.channels
        assert h == w == self.image_size
        assert cond_channels == self.cond_channels

        img = normalize_to_neg_one_to_one(img)
        seg = normalize_to_neg_one_to_one(seg) if seg is not None else None
        extra_cond = default(extra_cond, torch.zeros((b, self.extra_channels, h, w), device=self.device))
        times = torch.zeros((img.shape[0],), device=self.device).float().uniform_(0, 1)
        return self.p_losses(img, times, cond_img, seg, extra_cond, *args, **kwargs)

    def p_losses(self, x_start, times, cond_img, seg=None, extra_cond=None, noise=None, *args, **kwargs):
        noise = default(noise, lambda: torch.randn_like(x_start))

        # noise sample, if seg is not None, sample from seg.
        if seg is not None:
            x, log_snr = self.q_sample(x_start=seg, times=times, noise=noise)
        else:
            x, log_snr = self.q_sample(x_start=x_start, times=times, noise=noise)

        # predict and take gradient step
        model_out = self.model(torch.cat([x, extra_cond], dim=1), log_snr, cond_img)

        if self.pred_objective == 'v':
            padded_log_snr = right_pad_dims_to(x, log_snr)
            alpha, sigma = padded_log_snr.sigmoid().sqrt(), (-padded_log_snr).sigmoid().sqrt()
            target = alpha * noise - sigma * x_start
            # pred_x0 = alpha * x - sigma * model_out
            # true_x0 = alpha * x - sigma * target

        elif self.pred_objective == 'eps':
            target = noise

        elif self.pred_objective == 'x0':
            # Note: x_start in here is from -1 to 1
            target = (x_start + 1)/2

        if self.loss_type == 'l2':
            return F.mse_loss(model_out, target)
        elif self.loss_type == 'l1':
            return F.l1_loss(model_out, target)
        elif self.loss_type == 'l1+l2':
            return F.mse_loss(model_out, target) + F.l1_loss(model_out, target)
        elif self.loss_type == 'mean(l1, l2)':
            return (F.mse_loss(model_out, target) + F.l1_loss(model_out, target)) / 2
        else:
            return self.loss_type(model_out, target)

    @torch.no_grad()
    def sample(self, cond_img, extra_cond=None, verbose=True):
        b, c, h, w = cond_img.shape
        extra_cond = default(extra_cond, torch.zeros((b, self.extra_channels, h, w), device=self.device))
        return self.p_sample_loop((b, self.channels, self.image_size, self.image_size),
                                  cond_img,
                                  extra_cond=extra_cond, verbose=verbose)

    @torch.no_grad()
    def p_sample_loop(self, shape, cond_img, extra_cond, verbose=True):
        self.history = []
        img = torch.randn(shape, device=self.device)
        conditioning_features = self.model.extract_features(cond_img)
        steps = torch.linspace(1., 0., self.num_sample_steps + 1, device=self.device)

        for i in tqdm(range(self.num_sample_steps), desc='sampling loop time step', total=self.num_sample_steps,
                      disable=not verbose):
            times = steps[i]
            times_next = steps[i + 1]
            img = self.p_sample(img, conditioning_features, extra_cond, times, times_next)

        img.clamp_(-1., 1.)
        img = unnormalize_to_zero_to_one(img)
        return img

    @torch.no_grad()
    def p_sample(self, x, cond, extra_cond, time, time_next):
        batch, *_, device = *x.shape, x.device

        model_mean, model_variance = self.p_mean_variance(x=x, cond=cond, extra_cond=extra_cond, time=time,
                                                          time_next=time_next)

        if time_next == 0:
            return model_mean

        noise = torch.randn_like(x)
        return model_mean + sqrt(model_variance) * noise

    def p_sample_g(self, x, cond, extra_cond, time, time_next):
        batch, *_, device = *x.shape, x.device

        model_mean, model_variance = self.p_mean_variance(x=x, cond=cond, extra_cond=extra_cond, time=time,
                                                          time_next=time_next)

        if time_next == 0:
            return model_mean

        noise = torch.randn_like(x)
        return model_mean + sqrt(model_variance) * noise

    def p_mean_variance(self, x, cond, extra_cond, time, time_next):

        log_snr = self.log_snr(time)
        log_snr_next = self.log_snr(time_next)
        c = -expm1(log_snr - log_snr_next)

        squared_alpha, squared_alpha_next = log_snr.sigmoid(), log_snr_next.sigmoid()
        squared_sigma, squared_sigma_next = (-log_snr).sigmoid(), (-log_snr_next).sigmoid()

        alpha, sigma, alpha_next = map(sqrt, (squared_alpha, squared_sigma, squared_alpha_next))

        batch_log_snr = repeat(log_snr, ' -> b', b=x.shape[0])
        pred = self.model.sample_unet(torch.cat([x, extra_cond], dim=1),
                                      batch_log_snr, cond)

        if self.pred_objective == 'v':
            x_start = alpha * x - sigma * pred

        elif self.pred_objective == 'eps':
            x_start = (x - sigma * pred) / alpha

        elif self.pred_objective == 'x0':
            # raise NotImplementedError
            # due to we don't know x is normalized or not
            x_start = pred.tanh()
            # x_start = x

        x_start.clamp_(-1., 1.)
        self.history.append(x_start) # change to pred when generate cam

        model_mean = alpha_next * (x * (1 - c) / alpha + c * x_start)

        posterior_variance = squared_sigma_next * c

        return model_mean, posterior_variance


class ResCondGaussianDiffusion(CondGaussianDiffusion):
    def __init__(self, *args, **kwargs):
        super(ResCondGaussianDiffusion, self).__init__(*args, **kwargs)

    def forward(self, img, cond_img, seg=None, extra_cond=None, *args, **kwargs):
        b, channels, h, w = img.shape
        cond_channels = cond_img.shape[1]
        assert channels == self.channels
        assert h == w == self.image_size
        assert cond_channels == self.cond_channels

        # Here no need to normalize to [-1, 1] because the input is already normalized
        # img = normalize_to_neg_one_to_one(img)
        seg = normalize_to_neg_one_to_one(seg) if seg is not None else None
        extra_cond = default(extra_cond, torch.zeros((b, self.extra_channels, h, w), device=self.device))
        times = torch.zeros((img.shape[0],), device=self.device).float().uniform_(0, 1)
        return self.p_losses(img, times, cond_img, seg, extra_cond, *args, **kwargs)

    @torch.no_grad()
    def p_sample_loop(self, shape, cond_img, extra_cond, verbose=True):
        self.history = []
        img = torch.randn(shape, device=self.device)
        conditioning_features = self.model.extract_features(cond_img)
        steps = torch.linspace(1., 0., self.num_sample_steps + 1, device=self.device)

        for i in tqdm(range(self.num_sample_steps), desc='sampling loop time step', total=self.num_sample_steps,
                      disable=not verbose):
            times = steps[i]
            times_next = steps[i + 1]
            img = self.p_sample(img, conditioning_features, extra_cond, times, times_next)

        img.clamp_(-1., 1.)
        # Also we don't need to unnormalize the image because the output we need is [-1, 1]
        # img = unnormalize_to_zero_to_one(img)
        return img