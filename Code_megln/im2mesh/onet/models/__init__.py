import torch
import torch.nn as nn
from torch import distributions as dist
from im2mesh.onet.models import encoder_latent, decoder

# Encoder latent dictionary
encoder_latent_dict = {
    'simple': encoder_latent.Encoder,
}

# Decoder dictionary
decoder_dict = {
    'simple': decoder.Decoder,
    'cbatchnorm': decoder.DecoderCBatchNorm,
    'cbatchnorm2': decoder.DecoderCBatchNorm2,
    'batchnorm': decoder.DecoderBatchNorm,
    'cbatchnorm_noresnet': decoder.DecoderCBatchNormNoResnet,
}


class OccupancyNetwork(nn.Module):
    ''' Occupancy Network class.

    Args:
        decoder (nn.Module): decoder network
        encoder (nn.Module): encoder network
        encoder_latent (nn.Module): latent encoder network
        p0_z (dist): prior distribution for latent code z
        device (device): torch device
    '''

    def __init__(self, decoder, encoder=None, encoder_latent=None, p0_z=None,
                 device=None):
        super().__init__()
        if p0_z is None:
            p0_z = dist.Normal(torch.tensor([]), torch.tensor([]))

        self.decoder = decoder.to(device)

        if encoder_latent is not None:
            self.encoder_latent = encoder_latent.to(device)
        else:
            self.encoder_latent = None

        if encoder is not None:
            self.encoder = encoder.to(device)
        else:
            self.encoder = None

        self._device = device
        self.p0_z = p0_z

    def forward(self, p, inputs, sample=True, **kwargs):
        ''' Performs a forward pass through the network.

        Args:
            p (tensor): sampled points
            inputs (tensor): conditioning input
            sample (bool): whether to sample for z
        '''
        batch_size = p.size(0)
        xfg1, xfg2, x0, concate_out = self.encode_inputs(inputs)
        z = self.get_z_from_prior((batch_size,), sample=sample)
        out_concat, out_xfg1, out_xfg2, out, out_att = self.decode(p, z, z, z, z, xfg1, xfg2, x0, concate_out, **kwargs)
        return out_concat, out_xfg1, out_xfg2, out, out_att

    def compute_elbo(self, p, occ, inputs, **kwargs):
        ''' Computes the expectation lower bound.

        Args:
            p (tensor): sampled points
            occ (tensor): occupancy values for p
            inputs (tensor): conditioning input
        '''
        xfg1, xfg2, x0, concate_out = self.encode_inputs(inputs)
        q_z = self.infer_z(p, occ, concate_out, **kwargs)
        z = q_z.rsample()
        out_concat, out_xfg1, out_xfg2, out, out_att = self.decode(p, z, z, z, z, xfg1, xfg2, x0, concate_out, **kwargs)

        rec_error = -out_att.log_prob(occ).sum(dim=-1)
        kl = dist.kl_divergence(q_z, self.p0_z).sum(dim=-1)
        elbo = -rec_error - kl

        return elbo, rec_error, kl

    def encode_inputs(self, inputs):
        ''' Encodes the input.

        Args:
            input (tensor): the input
        '''

        if self.encoder is not None:
            c = self.encoder(inputs)
            # print(c)
        else:
            # Return inputs?
            c = torch.empty(inputs.size(0), 0)

        return c

    def decode(self, p, z_xfg1, z_xfg2, z_x0, z_concate_out, xfg1, xfg2, x0, concate_out, **kwargs):
        ''' Returns occupancy probabilities for the sampled points.

        Args:
            p (tensor): points
            z (tensor): latent code z
            c (tensor): latent conditioned code c
        '''

        out_concat, out_xfg1, out_xfg2, out, out_att = self.decoder(p, z_xfg1, z_xfg2, z_x0, z_concate_out, xfg1, xfg2, x0, concate_out, **kwargs)
        p_r_out_concat = dist.Bernoulli(logits=out_concat)
        p_r_out_xfg1 = dist.Bernoulli(logits=out_xfg1)
        p_r_out_xfg2 = dist.Bernoulli(logits=out_xfg2)
        p_r_out = dist.Bernoulli(logits=out)
        p_r_out_att = dist.Bernoulli(logits=out_att)
        return p_r_out_concat, p_r_out_xfg1, p_r_out_xfg2, p_r_out, p_r_out_att

    def infer_z(self, p, occ, c, **kwargs):
        ''' Infers z.

        Args:
            p (tensor): points tensor
            occ (tensor): occupancy values for occ
            c (tensor): latent conditioned code c
        '''
        if self.encoder_latent is not None:
            mean_z, logstd_z = self.encoder_latent(p, occ, c, **kwargs)
        else:
            batch_size = p.size(0)
            mean_z = torch.empty(batch_size, 0).to(self._device)
            logstd_z = torch.empty(batch_size, 0).to(self._device)

        q_z = dist.Normal(mean_z, torch.exp(logstd_z))
        return q_z

    def get_z_from_prior(self, size=torch.Size([]), sample=True):
        ''' Returns z from prior distribution.

        Args:
            size (Size): size of z
            sample (bool): whether to sample
        '''
        if sample:
            z = self.p0_z.sample(size).to(self._device)
        else:
            z = self.p0_z.mean.to(self._device)
            z = z.expand(*size, *z.size())

        return z

    def to(self, device):
        ''' Puts the model to the device.

        Args:
            device (device): pytorch device
        '''
        model = super().to(device)
        model._device = device
        return model
