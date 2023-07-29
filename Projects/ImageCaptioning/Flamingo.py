#!/usr/bin/env python
# coding: utf-8

# In[3]:


from __future__ import annotations
import torch
from torch import nn, einsum, tanh
from einops import repeat, rearrange
from einops_exts import rearrange_many
from transformers.configuration_utils import PretrainedConfig
from typing import List, Any, Dict, Tuple, Optional
from PIL import Image
from transformers import CLIPImageProcessor
from abc import ABC, abstractmethod
import contextlib
import logging
import torch.nn.functional as F
from transformers import PreTrainedModel
from transformers.models.clip.modeling_clip import CLIPVisionModel
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast
)
import requests


# In[4]:


@contextlib.contextmanager
def suppress_model_loading_warnings(suppress: bool = True):
    if suppress:
        logger = logging.getLogger('transformers.modeling_utils')
        level = logger.level
        logger.setLevel(logging.CRITICAL)
        yield
        logger.setLevel(level)
    else:
        yield


# In[5]:


def load_url(url: str):
    return Image.open(requests.get(url, stream=True).raw)


def load_image(path: str):
    return Image.open(path)


def unzip(l):
    return list(zip(*l))


class SquaredReLU(nn.Module):
    """ squared ReLU activation function"""
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        return torch.pow(torch.relu(x), 2)


def FeedForward(dim, mult=4, act='gelu'):
    """
    lucidrains implementation, slightly modified with the act parameter.
    """
    
    acts = dict(
        gelu=nn.GELU,
        sqrelu=SquaredReLU,
        relu=nn.ReLU
    )
    
    assert act in acts, f"act. can only be one of {acts.keys()}"
    
    inner_dim = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim, bias=False),
        acts[act](),
        nn.Linear(inner_dim, dim, bias=False)
    )


def get_common_prefix_length(x: torch.Tensor) -> int:
    # assuming that x is a matrix
    try:
        return (x[0] == x[1:]).all(dim=0).tolist().index(False)
    except ValueError:
        return x.size(1)


# In[6]:


class FlamingoConfig(PretrainedConfig):
    
    def __init__(
        self,
        lm: str = 'gpt2',
        clip_model_type: str = 'openai/clip-vit-base-patch32',
        dim: int = 1024,
        dim_visual: int = 768,
        xattn_every: int = 1,
        xattn_dim_head: int = 64,
        xattn_heads: int = 8,
        xattn_ff_mult: int = 4,
        xattn_act: str = 'gelu',
        resampler_depth: int = 6,
        resampler_dim_head: int = 64,
        resampler_heads: int = 8 ,
        resampler_num_latents: int = 64,
        resampler_num_time_embeds: int = 4,
        resampler_ff_mult: int = 4,
        resampler_act: str = 'gelu',
        freeze_language_model: bool = True,
        freeze_vision_model: bool = True,
        **kwargs
    ):
        """ Flamingo Configuration Class
        
        Args:
            lm (str): huggingface identifier of the language model. supported are 'gpt2' variations and 'facebook/opt-*'
            clip_model_type (str): huggingface identifier of the vision encoder.
            dim (int): LM embedding size
            dim_visual (int): Vision encoder embedding size
            xattn_every (int): frequency of interleaved gated xattn layers.
            xattn_dim_head (int): inner dim of xattn heads
            xattn_heads (int): number of attention heads in the xattn layers
            xattn_ff_mult (int): ?
            xattn_act (str): activation function to use in the xattn layers. Flamingo used 'sqrelu' in their paper.
            resampler_depth (int): number of attention layers in the perceiver resampler.
            resampler_dim_head: inner dim of resampler attention heads
            resampler_heads (int): number of attention heads in the resampler
            resampler_num_latents (int): number of learnable queries in the resampler
            resampler_num_time_embeds (int): ?
            resampler_ff_mult (int): ?
            resampler_act (str): activation function of the resampler. Flamingo used 'sqrelu' in their paper.
            freeze_language_model (bool): whether to freeze the language model or not.
            freeze_vision_model (bool): whether to freeze the vision model or not.
        """
        super().__init__(**kwargs)
        self.lm = lm
        self.clip_model_type = clip_model_type
        self.dim = dim
        self.dim_visual = dim_visual
        self.xattn_every = xattn_every
        self.xattn_dim_head = xattn_dim_head
        self.xattn_heads = xattn_heads
        self.xattn_ff_mult = xattn_ff_mult
        self.xattn_act = xattn_act
        self.resampler_depth = resampler_depth
        self.resampler_dim_head = resampler_dim_head 
        self.resampler_heads = resampler_heads 
        self.resampler_num_latents = resampler_num_latents
        self.resampler_num_time_embeds = resampler_num_time_embeds
        self.resampler_ff_mult = resampler_ff_mult
        self.resampler_act = resampler_act
        self.freeze_language_model = freeze_language_model
        self.freeze_vision_model = freeze_vision_model


# In[7]:


class FlamingoProcessor:
    """ 
    FlamingoProcessor offers functions to preprocess the raw data (images and text).
    Wrapper around a transformer GPT-2 tokenizer and a clip processor.
    """
    
    vision_processor: CLIPImageProcessor

    def __init__(
        self,
        config: FlamingoConfig,
        use_fast: bool = True,
        eoc_token: str = '<EOC>'
    ):
        """
        Args:
            config (FlamingoConfig): pass the same FlamingoConfig as used to initialize the FlamingoModel.
            use_fast (bool): whether to use the fast tokenizer implementations from huggingface.
            eoc_token (str): string representation of the token to add.
        """
        self.config = config
        self.eoc_token = eoc_token
        self.vision_processor = CLIPImageProcessor.from_pretrained(config.clip_model_type) #type: ignore
        
        if config.lm.startswith('gpt2'):
            if use_fast:
                from transformers import GPT2TokenizerFast

                self.tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
            else:
                from transformers import GPT2Tokenizer

                self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        elif config.lm.startswith('facebook/opt'):
            from transformers import AutoTokenizer
            
            self.tokenizer = AutoTokenizer.from_pretrained('facebook/opt-30b', use_fast=use_fast)
        elif config.lm.startswith('bigscience/bloom-560m'):
            from transformers import AutoTokenizer
            
            self.tokenizer = AutoTokenizer.from_pretrained('bigscience/bloom-560m', use_fast=use_fast)
        
        self.tokenizer.add_bos_token = True
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.add_tokens(self.eoc_token)

        # find the start token for "<image>". " <" is 1279, "<" is 27
        # the encoded "<" token-id is different if there is a preceding whitespace.
        #        with ws    without
        # gpt-2:  1279         27
        # opt:   28696      51552
        self.leq_ids = [
            self.tokenizer.encode("<")[-1],
            self.tokenizer.encode(" <")[-1]
        ]

    def encode_text(
        self,
        text: str | List[str],
        device: torch.device | None = None,
        max_length=None,
        length=None,
        return_tensors='pt',
        return_attention_mask=True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        if length is not None:
            result = self.tokenizer(
                text,
                return_tensors=return_tensors,
                return_attention_mask=return_attention_mask,
                padding='max_length',
                truncation=True,
                max_length=length)
        elif max_length is None:
            result = self.tokenizer(
                text,
                return_tensors=return_tensors, 
                padding=True)
        else:
            result = self.tokenizer(
                text,
                return_tensors=return_tensors,
                return_attention_mask=return_attention_mask,
                padding=True,
                truncation=True,
                max_length=max_length)
            
            
        media_locs = self.get_media_locations(result.input_ids)

        return result.input_ids.to(device), media_locs.to(device), result.attention_mask.to(device)
    
    def prepare_caption(self, caption: str) -> str:
        # <BOS> token is added automatically by the tokenizer.
        # <EOS> token is not.
        return "<image>" + caption + self.eoc_token + self.tokenizer.eos_token
            
    def prepare_captions(self, captions: List[str]) -> List[str]:
        """preparation function for the conceptual captions dataset. """
        return [self.prepare_caption(c) for c in captions]
        
    def _remove_tags(self, text: str) -> str:
        for s in ('<image>', self.tokenizer.eos_token, self.eoc_token, self.tokenizer.pad_token):
            text = text.replace(s, '')
        return text.strip()
    
    def remove_tags(self, text: str | List[str]) -> str | List[str]:
        if isinstance(text, str):
            return self._remove_tags(text)
        else:
            return [self._remove_tags(t) for t in text]
    
    def get_media_locations(self, input_ids: torch.Tensor) -> torch.Tensor:
        return torch.stack([(input_ids == leq_id) for leq_id in self.leq_ids]).sum(0)
    
    def preprocess_images(self, images: List[Image.Image]) -> torch.Tensor:
        """
        :param images: a list of PIL image instances
        :return: Tensor of shape [n_images, width, height, depth]
        """
        return self.vision_processor(images=images, return_tensors="pt", padding=True)

    def __call__(
        self, 
        images: Image.Image | List[Image.Image] | torch.Tensor | List[torch.Tensor] | None = None, 
        text: str | List[str] | None = None, 
        device: torch.device | None = None
    ):
        result = {}
        
        if images is not None:
            result['pixel_values'] = self.vision_processor(images=images, return_tensors='pt', padding=True)['pixel_values'].to(device)
            
        if text is not None:
            input_ids, media_locations, attention_mask = self.encode_text(text, device=device)
            result['input_ids'] = input_ids
            result['media_locations'] = media_locations
            result['attention_mask'] = attention_mask

        return result


# In[8]:


class MaskedCrossAttention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_visual,
        dim_head=64,
        heads=8,
        n_visual=64
    ):
        """
        :param dim:      d_token, d_visual  dimensionality of language- and visual tokens
        :param dim_head: dimensionality of the q, k, v vectors inside one attention head
        :param heads:   number of attention heads
        """
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        self.n_visual = n_visual
        inner_dim = dim_head * heads

        self.norm = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim_visual, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(
        self,
        y: torch.Tensor,
        media_locations: torch.Tensor,
        visual_features: torch.Tensor,
        previous_kv=None,
        output_kv=False
    ):
        """This has the same inputs as the GatedCrossAttentionBlock

        Args:
            y (FloatTensor):
                language features (n_batch, n_token, d_token)
            visual_features (FloatTensor, optional):
                visual features   (n_batch, n_images, n_queries, d_visual). Defaults to None.
            media_locations (BoolTensor, optional):
                boolean matrix of shape: (n_batch, n_token). Defaults to None.
                Indicates the positions of <image> tokens in the input sequence y (more precisely, the position of '<')
            previous_kv (Tuple, optional):
                tuple of previous keys and values. Passed when caching is used during text generation.
                Defaults to None.
            output_kv (bool, optional):
                whether to return the keys and values. Defaults to False.

        Returns:
            FloatTensor: Tensor (n_batch, n_token, d_token)
        """
        n_batch, n_media = visual_features.shape[:2]
        n_batch_y, n_token, d_token = y.shape
        n_heads = self.heads

        # LayerNorm
        y = self.norm(y)

        # 2. compute the queries from the text tokens:
        q = self.to_q(y)
        q = q * self.scale

        # 3. compute the keys and values from the visual tokens:
        if previous_kv is None:
            # flatten, so t is #images, n is #visual features per image.
            # Now there is only one set of visual features per # sentence.
            visual_features = rearrange(visual_features, 'b t n d -> b (t n) d')

            k, v = self.to_kv(visual_features).chunk(2, dim=-1)
            q, k, v = rearrange_many((q, k, v), 'b n (h d) -> b h n d', h=n_heads)
        else:
            # visual_features can be ignored, k, v already computed
            k, v = previous_kv
            n_media = k.size(2) // self.n_visual
            q = rearrange(q, 'b n (h d) -> b h n d', h=n_heads)

        # 5. compute the attention scores from the queries and keys:
        sim = einsum('... i d, ... j d -> ... i j', q, k)

        text_time = media_locations.cumsum(dim=-1) # at each boolean of True, increment the time counter (relative to media time)
        
        # >> David
        # this needs to be adjusted if caching is used.
        # text_time has shape (n_batch, n_token)
        if previous_kv is not None:
            text_time = text_time[:, -n_token:]
            assert text_time.shape == y.shape[:2]
        
        media_time = torch.arange(n_media, device=y.device) + 1
        # >> David:
        # side note: here, text tokens attend to ALL previous visual tokens. If We only want to attend to the
        # one image coming before in the text (like in the flamingo paper),
        # we need to change >= to == at the line where 'text_to_media_mask' is created.
        text_to_media_mask = rearrange(text_time, 'b i -> b 1 i 1') == repeat(media_time, 'j -> 1 1 1 (j m)', m=self.n_visual)
        sim = sim.masked_fill(~text_to_media_mask, -torch.finfo(sim.dtype).max)

        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        alphas = sim.softmax(dim=-1)
        
        # a bug and an update from lucidrains: 
        # any text without a preceding media needs to have attention zeroed out
        text_without_media_mask = text_time == 0
        text_without_media_mask = rearrange(text_without_media_mask, 'b i -> b 1 i 1')
        alphas = alphas.masked_fill(text_without_media_mask, 0.)

        out = einsum('... i j, ... j d -> ... i d', alphas, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        
        conditioned_tokens = self.to_out(out)

        if output_kv:
            return conditioned_tokens, (k, v)
        else:
            return conditioned_tokens, None
            


class GatedCrossAttentionBlock(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_visual,
        dim_head=64,
        heads=8,
        ff_mult=4,
        act='gelu',
        n_visual=64
    ):
        """
        :param dim:      d_token, d_visual
        :param dim_head: dimensionality of q, k, v inside the attention head
        :param heads:    number of attention heads
        :param ff_mult:  factor for the number of inner neurons in the ffw layer
        """
        super().__init__()
        self.attn = MaskedCrossAttention(dim=dim, dim_visual=dim_visual, dim_head=dim_head, heads=heads, n_visual=n_visual)
        self.alpha_attn = nn.Parameter(torch.tensor([0.]))

        self.ffw = FeedForward(dim, mult=ff_mult, act=act)
        self.alpha_ffw = nn.Parameter(torch.tensor([0.]))

    def forward(
        self, 
        y: torch.Tensor, 
        visual_features: torch.Tensor, 
        media_locations: torch.Tensor, 
        previous_kv=None, 
        output_kv=False
    ):
        """
        :param y:           (n_batch, n_tokens, d_token) - language features from previous LM layer
        :param media:       (n_batch, n_media, n_queries, dim) - visual features, encoded by perceiver resample
        :param media_locations:  (n_batch, n_tokens) - boolean matrix, indicating start positions of <image> tokens
        :return:
        """
        if previous_kv is None:
            assert visual_features.ndim == 4
        shape_before = y.shape
        
        # kv will be None if output_kv=False
        attn_out, kv = self.attn(y, media_locations, visual_features, previous_kv=previous_kv, output_kv=output_kv)
        y = y + tanh(self.alpha_attn) * attn_out
        assert y.shape == shape_before        
        y = y + tanh(self.alpha_ffw) * self.ffw(y)
        assert y.shape == shape_before        
        return y, kv


class ModifiedLMBlock(nn.Module):
    """
    A block that wraps a gated cross-attention layer, followed by a LM layer.
    We replace the original layers in the LM with these at a certain frequency
    to introduce the xattn layer. This layer mimics the functionality and behavior 
    of the underlying LM block. This way, the LM can be used in the same way as before,
    and we can do the conditioning without altering the LM implementation.
    
    One drawback of this approach is that we cannot pass the visual features to forward()
    directly, but instead we need to pass them before the actual forward pass, via a 
    side-channel, which is the condition() method. In addition, when use_cache is used,
    the cached keys and values for the xattn layers need to be retrieved separately from
    the kv_output property.
    
    (!) This implementation works with GPT-2 and OPT layers, but hasn't been tested with other LMs yet.
    """
    
    def __init__(self, lm_block, **kwargs):
        super().__init__()
        
        self.xattn_block = GatedCrossAttentionBlock(**kwargs)
        self.lm_block = lm_block
        self.visual_features = None
        self.media_locations = None
        self.xattn_layer_past = None
        self.kv_output = None
        
    def condition(self, visual_features: torch.Tensor, media_locations: torch.Tensor, xattn_layer_past=None) -> None:
        """
        conditioning. Called from outside of the LM before passing the text input to the LM.
        This way, the gated cross-attention layers get informed about the visual input
        without the need to pipe the visual input through the LM forward() function.
        
        xattn_layer_past can contain the cached cross-attention keys and values (computed
        from the visual input). Passing them is useful to speed up the autoregressive text
        generation where the keys and values will be the same for every word, since the 
        visual input doesn't change.
        If both visual_features and xattn_layer past are passed, visual_features will be 
        ignored in the xattn layers.
        """
        self.visual_features = visual_features 
        self.media_locations = media_locations
        self.xattn_layer_past = xattn_layer_past
        
    def forward(
        self,
        hidden_states: Tuple[torch.Tensor] | None,
        use_cache: bool | None = False,
        **kwargs
    ):
        """
        This forward function mimics forward() of GPT2Block, so it has the same input and output.
        """
        
        # pass through xattn
        hidden_states, kv = self.xattn_block(
            y=hidden_states, 
            visual_features=self.visual_features, 
            media_locations=self.media_locations,
            previous_kv=self.xattn_layer_past,
            output_kv=use_cache
        )
        self.kv_output = kv
        
        # pass through original LM layer
        return self.lm_block(hidden_states, use_cache=use_cache, **kwargs)
        
    


# In[9]:


class FlamingoBaseModel(ABC, PreTrainedModel):
    """ 
    abstract class, which is inherited by FlamingoGPT2 and FlamingoOPT.
    This class provides the core functionalities of Flamingo: the forward() function,
    setting up the resampler and hijacking the LM layers with GatedXAttn layers.
    """

    config: FlamingoConfig
    vision_encoder: CLIPVisionModel
    resampler: PerceiverResampler
    lm: PreTrainedModel
    lm_head: nn.Linear

    config_class = FlamingoConfig

    def __init__(self, config: FlamingoConfig, suppress_warnings=True):
        assert isinstance(config, FlamingoConfig)
        super().__init__(config)
        
        with suppress_model_loading_warnings(suppress_warnings):
            self.vision_encoder = CLIPVisionModel.from_pretrained(config.clip_model_type) # type: ignore

        self.resampler = PerceiverResampler(
            dim=config.dim_visual,
            depth=config.resampler_depth,
            dim_head=config.resampler_dim_head,
            heads=config.resampler_heads,
            num_latents=config.resampler_num_latents,
            num_time_embeds=config.resampler_num_time_embeds,
            ff_mult=config.resampler_ff_mult,
            act=config.resampler_act
        )

    def _init_layers(self, lm_layers: nn.ModuleList):
        """ 
        call during init of the subclass.
        careful, this method will modify the LM layers!
        """
        for i, lm_layer in enumerate(lm_layers):
            if i % self.config.xattn_every != 0: 
                continue

            lm_layers[i] = ModifiedLMBlock(
                lm_layer,
                dim=self.config.dim,
                dim_visual=self.config.dim_visual,
                dim_head=self.config.xattn_dim_head,
                heads=self.config.xattn_heads,
                ff_mult=self.config.xattn_ff_mult,
                act=self.config.xattn_act,
                n_visual=self.config.resampler_num_latents
            )
            
    @abstractmethod
    def get_modified_layers(self) -> List[ModifiedLMBlock]:
        raise NotImplementedError
            
    def freeze_vm(self):
        """freeze vision model """
        for param in self.vision_encoder.parameters():
            param.requires_grad = False

    def freeze_lm(self):
        """ freeze weights of the language model.

        (!) does not freeze token embedding matrix and gated xattn layers
        """

        for param in self.lm.parameters():
            param.requires_grad = False

        # lm_head shares weights with the embeddings so no need to unfreeze that as well
        self.lm.get_input_embeddings().weight.requires_grad = True

        for xattn in self.get_modified_layers():
            for param in xattn.xattn_block.parameters():
                param.requires_grad = True

    def unfreeze_lm(self):
        for param in self.lm.parameters():
            param.requires_grad = True

    def state_dict_trainable(self) -> Dict[str, torch.Tensor]:
        """ include weights in the state dict if they have requires_grad = True"""

        trainable_param_names = [
            w for w, t in self.named_parameters() if t.requires_grad]
        return {k: v for k, v in self.state_dict().items() if k in trainable_param_names}

    def parameters_trainable(self):
        """Access the trainable parameters, e.g. useful for the optimizer and gradient clipping. 

        example: optimizer = AdamW(model.parameters_trainable(), lr=args.lr)
        make sure to call freeze_lm() first! 
        """
        return filter(lambda p: p.requires_grad, self.parameters())
    
    def encode_resample_visuals(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """pass pixel values through vision encoder and perceiver resampler.

        Args:
            pixel_values (torch.Tensor): accepted shapes:
                (N c h w)       one batch, multiple images
                (b N c h w)     multiple batches, multiple images
                (b N T c h w)   multiple batches, multiple images, multiple frames

        Returns:
            (torch.Tensor): shape (b N q d)
        """

        if pixel_values.ndim == 4:            
            # (N c h w)
            b, N, T = 1, pixel_values.size(0), 1
        
        elif pixel_values.ndim == 5:       
            # (b N c h w)
            b, N, T = *pixel_values.shape[:2], 1
            pixel_values = rearrange(pixel_values, 'b N c h w -> (b N) c h w')

        elif pixel_values.ndim == 6:         
            # (b N T c h w) -> (b N T v d)
            b, N, T = pixel_values.shape[:3]
            pixel_values = rearrange(pixel_values, 'b N T c h w -> (b N T) c h w')
        else:
            raise ValueError('pixel_values must have ndim 5 or 6!')

        with torch.no_grad():
            visual_features = self.vision_encoder(pixel_values).last_hidden_state         # (b N T) v d

        # perceiver resampler
        # (only need to do if kv of the xattn layers were not calculated yet.)
        # resample visual features ((b N T) v d) -> (b N T q d)
        visual_features = rearrange(visual_features, '(b N T) v d -> (b N) T v d', b=b, N=N, T=T)
        visual_features = self.resampler(visual_features)

        # T is gone at this point
        visual_features = rearrange(visual_features, '(b N) q d -> b N q d', b=b, N=N)
        
        return visual_features
        
    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        media_locations: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        visual_features: torch.Tensor | None = None,
        head_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        use_cache: bool = False,
        past_key_values: tuple | None = None,
        return_dict: bool = True,
        labels: torch.Tensor | None = None,
        loss_reduction: str = 'mean',
        **kwargs
    ) -> CausalLMOutputWithPast:
        """Flamingo forward pass

        Most of the parameters are inspired by huggingface language model implementations, so this doc may be informative:
        https://huggingface.co/docs/transformers/model_doc/gpt2#transformers.GPT2Model.forward

        Args:
            input_ids (Tensor | None):         shape (n_batch, n_tokens). the tokenized input text
            attention_mask (Tensor | None):    shape (n_batch, n_tokens). 
                Mask as produced by the tokenizer. Required when a batch of input strings are tokenized and thus padded at the end.
                Then this will indicate the locations of 'real' tokens vs. the location of 'pad' tokens.
            media_locations (Tensor | None):   shape (n_batch, n_tokens).
                indicates the locations of the starts of the <image> tags beginning, i.e. the location of the token representing '<'
            pixel_values (Tensor | None):    shape (b N T c h w). Optional.
            visual_features (Tensor | None):         shape (b N q d). Optional.
                If pixel_values already have been passed through encode_resample_visuals(), 
                you can pass the resampled visual embeddings via this parameter.
                If provided, pixel_values will be ignored
            head_mask (Tensor | None): TODO
            inputs_embeds (Tensor | None): TODO
            use_cache (bool): whether to return the inner keys and values. Used to speed up text generation at inference. defaults to False
            past_key_values (tuple): tuple of past_key_values of (1) the xattn layers (2) the language model
            return_dict (bool): Whether to return a dictionary. Right now, only dicts are supported, so this must be set to True. Defaults to True.
            labels (Tensor): 
                It is possible to pass the exact value as input_ids also as labels. If present, the output will contain a CE loss of the next token prediction.
                optional, defaults to None
            **kwargs

        Returns:
            (CausalLMOutputWithPast): an object containing all the useful stuff. Refer to hf documentation.

        """

        # sanity check
        assert return_dict, "can only use return_dict=True at the moment!"
        assert (input_ids is None) != (inputs_embeds is None), "you must pass either input_ids or inputs_embeds!"

        # find the input shape
        batch_size, seq_length = input_ids.shape[:2] if input_ids is not None else inputs_embeds.shape[:2]
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        xattn_past_key_values = None if past_key_values is None else past_key_values[0]
        lm_past_key_values = None if past_key_values is None else past_key_values[1]
        
        if visual_features is None:
            if xattn_past_key_values is None and pixel_values is not None:
                # extract from pixels
                assert pixel_values.size(0) == batch_size, \
                    "pixel_values must have the same batch size as the textual input!"
                
                visual_features = self.encode_resample_visuals(pixel_values)
                
            else:
                # we don't need visual_features is past is defined.
                # use dummy values, since are only required for the shape
                # visual_embedings shape (b N q d)
                visual_features = torch.zeros(
                    (batch_size, 1, self.config.resampler_num_latents, self.config.dim_visual),
                    dtype=torch.float32,
                    device=device
                )

        if media_locations is None:
            media_locations = torch.zeros(size=(batch_size, seq_length), dtype=torch.int, device=device)

        # condition xattn layers
        for i, xattn in enumerate(self.get_modified_layers()):
            layer_past = None if xattn_past_key_values is None else xattn_past_key_values[i]
            xattn.condition(visual_features, media_locations, layer_past)

        # pass through LM
        out: BaseModelOutputWithPast = self.lm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            past_key_values=lm_past_key_values,
            return_dict=True,
            **kwargs
        )

        logits: torch.Tensor = self.lm_head(out.last_hidden_state)

        # collect the past_key_values from the xattn layers
        if use_cache:
            xattn_past_key_values = []
            for modified_layer in self.get_modified_layers():
                xattn_past_key_values.append(modified_layer.kv_output)

        loss = None
        if labels is not None:
            # loss function calculation, inspired by hf implementations
            # Shift so that tokens < n predict n
            # logits shape (batch, seq_length, #words)
            shift_logits = logits[..., :-1, :].contiguous()
            # labels shape (batch, seq_length)
            shift_labels = labels[..., 1:].contiguous()

            # Flatten the tokens
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)),
                                   shift_labels.view(-1), reduction=loss_reduction)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=(tuple(xattn_past_key_values), out.past_key_values) if use_cache else None,
            hidden_states=out.hidden_states,
            attentions=out.attentions,
        )


# In[10]:


class FlamingoGPT2(FlamingoBaseModel):
    config: FlamingoConfig
    config_class = FlamingoConfig

    def __init__(self, config: FlamingoConfig):
        from transformers import GPT2LMHeadModel, GPT2Model
        assert config.lm.startswith('gpt')
        super().__init__(config)

        base_lm: GPT2LMHeadModel = GPT2LMHeadModel.from_pretrained(config.lm)  # type: ignore
        
        base_lm.resize_token_embeddings(base_lm.config.vocab_size + 1)
        self.lm: GPT2Model = base_lm.transformer
        self.lm_head = base_lm.lm_head
        self._init_layers(self.lm.h)
        
    def get_modified_layers(self):
        if self.config.xattn_every == 1:
            return self.lm.h
        return filter(lambda layer: isinstance(layer, ModifiedLMBlock), self.lm.h)



# In[11]:


class FlamingoOPT(FlamingoBaseModel):
    config: FlamingoConfig
    config_class = FlamingoConfig

    def __init__(self, config: FlamingoConfig):
        from transformers import OPTForCausalLM, OPTModel
        assert config.lm.startswith('facebook/opt')
        super().__init__(config)

        base_lm: OPTForCausalLM = OPTForCausalLM.from_pretrained(config.lm)  # type: ignore

        base_lm.resize_token_embeddings(base_lm.config.vocab_size + 1)
        self.lm: OPTModel = base_lm.model
        self.lm_head = base_lm.lm_head
        self._init_layers(self.lm.decoder.layers)
        
    def get_modified_layers(self):
        if self.config.xattn_every == 1:
            return self.lm.decoder.layers
        return filter(lambda layer: isinstance(layer, ModifiedLMBlock), self.lm.decoder.layers)
    



class FlamingoBLOOM(FlamingoBaseModel):
    config: FlamingoConfig
    config_class = FlamingoConfig

    def __init__(self, config: FlamingoConfig):
        from transformers import BloomForCausalLM, BloomModel
        assert config.lm.startswith('bigscience/bloom-560m')
        super().__init__(config)

        base_lm: BloomForCausalLM = BloomForCausalLM.from_pretrained(config.lm)  # type: ignore

        base_lm.resize_token_embeddings(base_lm.config.vocab_size + 1)
        self.lm: BloomModel = base_lm.transformer
        self.lm_head = base_lm.lm_head
        self._init_layers(self.lm.h)
        
    def get_modified_layers(self):
        if self.config.xattn_every == 1:
            return self.lm.h
        return filter(lambda layer: isinstance(layer, ModifiedLMBlock), self.lm.h)



# In[12]:


class FlamingoModel(PreTrainedModel):
    """wrapper class for a FlamingoBase decending model (FlamingoGPT2 or FlamingoOPT)

    A generic flamingo interface that is independent of the underlying LM. Most of the methods are just forwarding to the actual model.
    This class implements prepare_inputs_for_generation() and reorder_cache(), which are required to utilize hf text generation methods.
    It also has a generate_captions() utility that can be used to create a caption for an image.
    """
    config: FlamingoConfig
    config_class = FlamingoConfig

    # key = prefix of an existing pretrained huggingface transformer language model
    # value = Flamingo class for the respective language model
    _LANGUAGE_MODEL_VERSIONS = {
        'gpt2': FlamingoGPT2,
        'facebook/opt': FlamingoOPT,
        'bigscience/bloom': FlamingoBLOOM
    }
    
    _keys_to_ignore_on_load_missing = [r"flamingo.vision_encoder"]

    def __init__(self, config: FlamingoConfig, model_class: type | None = None):
        """constructor.

        Args:
            config (FlamingoConfig): 
                config for the flamingo model.
            model_class (Optional[type], optional): 
                optionally use a custom class that inherits FlamingoBaseModel. 
                If none, it will choose FlamingoGPT2 or FlamingoOPT based on the FlamingoConfig. Defaults to None.
        """
        super().__init__(config)

        if model_class is None:
            model_class = self._find_flamingo_class(config.lm)
        self.flamingo: FlamingoBaseModel = model_class(config)
        
        if config.freeze_language_model:
            self.freeze_lm()

        if config.freeze_vision_model:
            self.freeze_vm()

    @classmethod
    def is_lm_supported(cls, lm_id: str) -> bool:
        return any(lm_id.startswith(prefix) for prefix in cls._LANGUAGE_MODEL_VERSIONS.keys())

    @classmethod
    def _find_flamingo_class(cls, language_model_id: str):
        for prefix, flamingo_class in cls._LANGUAGE_MODEL_VERSIONS.items():
            if language_model_id.startswith(prefix):
                return flamingo_class
        raise ValueError(f'unsupported language model {language_model_id}')

    def parameters_trainable(self):
        """Access the trainable parameters, e.g. useful for the optimizer and gradient clipping. 

        example: optimizer = AdamW(model.parameters_trainable(), lr=args.lr)
        make sure to call freeze_lm() first! 
        """
        return self.flamingo.parameters_trainable()

    def freeze_vm(self):
        self.flamingo.freeze_vm()

    def freeze_lm(self):
        self.flamingo.freeze_lm()

    def unfreeze_lm(self):
        self.flamingo.unfreeze_lm()

    def state_dict_trainable(self):
        return self.flamingo.state_dict_trainable()

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        media_locations: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        visual_features: torch.Tensor | None = None,
        head_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        use_cache: bool = False,
        past_key_values: tuple | None = None,
        return_dict: bool = True,
        labels: torch.Tensor | None = None,
        loss_reduction: str = 'mean',
        **kwargs
    ) -> CausalLMOutputWithPast:

        return self.flamingo(
            input_ids=input_ids,
            attention_mask=attention_mask,
            media_locations=media_locations,
            pixel_values=pixel_values,
            visual_features=visual_features,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            past_key_values=past_key_values,
            return_dict=return_dict,
            labels=labels,
            loss_reduction=loss_reduction,
            **kwargs
        )

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.Tensor,
        media_locations: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        visual_features: torch.Tensor | None = None,
        past=None,
        past_key_values=None,
        **kwargs
    ) -> Dict[str, Any]:
        """ hf specific function. Overridden from PreTrainedModel for text generation purposes.

        for beam search, input_ids is replicated times the number of beams. 
        I.e., batch_size' = batch_size * num_beams. 
        This function replicates also the visual_features and media_locations accordingly.

        if use_cache is used, past is not None, then only the last column will be passed as input_ids.
        TODO was `past` renamed to `past_key_values` in transformers 4.26?
        """

        if pixel_values is not None:
            n_inputs = input_ids.shape[0]
            n_visual = pixel_values.shape[0]

            if n_inputs != n_visual:
                assert n_inputs % n_visual == 0
                pixel_values = repeat(
                    pixel_values, 'n ... -> (n m) ...', m=n_inputs // n_visual)

        if visual_features is not None:
            n_inputs = input_ids.shape[0]
            n_visual = visual_features.shape[0]

            if n_inputs != n_visual:
                assert n_inputs % n_visual == 0
                visual_features = repeat(
                    visual_features, 'n ... -> (n m) ...', m=n_inputs // n_visual)

        if media_locations is not None:
            n_inputs = input_ids.shape[0]
            n_inputs_media = media_locations.shape[0]

            if n_inputs != n_inputs_media:
                assert n_inputs % n_inputs_media == 0
                media_locations = repeat(
                    media_locations, 'n ... -> (n m) ...', m=n_inputs // n_inputs_media)

        if past_key_values is not None or past is not None:
            input_ids = input_ids[:, -1:]

        return dict(
            input_ids=input_ids,
            past_key_values=past_key_values if past_key_values is not None else past,
            media_locations=media_locations,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            visual_features=visual_features,
            **kwargs
        )

    def _reorder_cache(self, past, beam_idx):
        """ hf specific function. Overridden from PreTrainedModel.

        this is required for beam search in combination with use_cache.

        Args: 
            past is a tuple of past_key_values of the xattn layers, and of the LM layers.
            beam_idx: index of the beam
        """
        xattn_past, lm_past = past

        xattn_past_beam = tuple(
            tuple(past_state.index_select(0, beam_idx.to(past_state.device))
                  for past_state in layer_past)
            for layer_past in xattn_past
        )

        lm_past_beam = tuple(
            tuple(past_state.index_select(0, beam_idx.to(past_state.device))
                  for past_state in layer_past)
            for layer_past in lm_past
        )

        return xattn_past_beam, lm_past_beam

    @torch.no_grad()
    def generate_captions(
        self,
        processor: FlamingoProcessor,
        pixel_values: torch.Tensor | None = None,
        images: Image.Image | List[Image.Image] | None = None,
        prompt: str = "<image>",
        max_length: int = 150,
        num_beams: int = 1,
        device: torch.device | None = None,
        **kwargs
    ):
        """
        helper utility for image captioning.
        prompt is replicated for all batches.
        """
        if device is None:
            device = self.device

        if images is not None:
            assert pixel_values is None, "you can only pass either images or visual features to generate_captions()!"

            if isinstance(images, Image.Image):
                images = [images]

            pixel_values = processor(images=images, device=device)['pixel_values']

        assert pixel_values is not None, "you must pass either images or visual features to generate_captions()!"

        batch_size = pixel_values.size(0)
        input_ids, media_locations, attention_mask = processor.encode_text(
            prompt, device)

        input_ids = repeat(input_ids[0], 'l -> n l', n=batch_size)
        media_locations = repeat(media_locations[0], 'l -> n l', n=batch_size)
        attention_mask = repeat(attention_mask[0], 'l -> n l', n=batch_size)

        out_ids = self.generate(
            inputs=input_ids,
            media_locations=media_locations,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            num_beams=num_beams,
            early_stopping=True,
            use_cache=True,
            bos_token_id=self.flamingo.lm.config.bos_token_id,
            eos_token_id=self.flamingo.lm.config.eos_token_id,
            pad_token_id=self.flamingo.lm.config.eos_token_id,
            max_length=max_length,
            **kwargs
        )

        captions = processor.tokenizer.batch_decode(
            out_ids, skip_special_tokens=True)
        captions = [processor.remove_tags(t) for t in captions]
        return captions

    @torch.no_grad()
    def score_sequences(
        self,
        input_ids: torch.Tensor,
        media_locations: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor | None = None,
        visual_features: torch.Tensor | None = None,
        k: int = 100000,
    ) -> torch.Tensor:
        """

        EXPERIMENTAL

        This method can be used for zero-shot classification.
        Given a batch of tokenized sentences, it computes the log-prob over each sample.

        inspired by ALBEF:
            https://github.com/salesforce/ALBEF/blob/b9727e43c3040491774d1b22cc27718aa7772fac/models/model_vqa.py#L149

        To improve efficiency, the implementation works like this:
        (1) find the longest common prefix over all sequences.
        (2) pass the prefix once and obtain the attention keys and values for LM and xattn layers
        (3) based on the likelihood for the next token, filter the top-k sequences for the next steps.
        (4) repeat keys and values for all top-k sequences
        (5) pass the top-k sequences to the model. use cached kv
        (6) compute the log-prob from the remaining parts and use as a score.
            For all sequences that didn't make it to the top-k, set the score to -inf

        TODO method fails when all sequences are equal

        Args:
            input_ids (torch.Tensor):           (b L)
            media_locations (torch.Tensor):     (b L)
            attention_mask (torch.Tensor):      (b L)
            pixel_values (torch.Tensor)         (N c h w)
            visual_features (torch.FloatTensor):    (N q d)
                These are the resampled visual embeddings 
                (!) the visual features are treated as the same for the complete batch of sentences

        Returns:
            torch.Tensor: log-probs for the batch of input sequences
                Tensor of shape [b], dtype torch.float 
        """

        assert visual_features.ndim == 3, f"visual_features must have shape (N q d), but has {visual_features.ndim} dimensions!"

        n_choices = input_ids.size(0)
        n_reuse = get_common_prefix_length(input_ids)
        k = min(k, n_choices)

        # first, pass the complete prefix and compute the hidden states.
        out = self.flamingo(
            input_ids=input_ids[:1, :n_reuse],
            media_locations=media_locations[:1, :n_reuse],
            attention_mask=attention_mask[:1, :n_reuse],
            pixel_values=pixel_values.unsqueeze(0) if pixel_values is not None else None,           # add batch_size of 1    
            visual_features=visual_features.unsqueeze(0) if visual_features is not None else None,  # add batch_size of 1
            use_cache=True,
        )

        next_tokens = input_ids[:, n_reuse]
        next_token_logits = out.logits[0, -1, :].index_select(0, next_tokens)
        topk_indices = next_token_logits.topk(k).indices

        # extend past_key_values to all sequences
        xattn_past_key_values = [
            tuple(repeat_many(kv, "1 ... -> b ...", b=k))
            for kv in out.past_key_values[0]
        ]
        lm_past_key_values = [
            (
                repeat(keys, "1 ... -> b ...", b=k)[:, :, :-1, :],
                repeat(vals, "1 ... -> b ...", b=k)[:, :, :-1, :],
            )
            for keys, vals in out.past_key_values[1]
        ]

        past_key_values = (xattn_past_key_values, lm_past_key_values)

        # then pass all choice sequences individually.
        choice_input_ids = input_ids[topk_indices, n_reuse - 1:]
        choice_media_locations = media_locations[topk_indices]
        choice_attention_mask = attention_mask[topk_indices]

        # at this point, we don't need the visual features anymore, since they have already been passed through
        # the perceiver resampler and the keys and values for them have been precomputed in the xattn layers.
        out2 = self.flamingo(
            input_ids=choice_input_ids,
            media_locations=choice_media_locations,
            attention_mask=choice_attention_mask,
            pixel_values=None,
            visual_features=None,
            past_key_values=past_key_values,
            labels=choice_input_ids,
            loss_reduction="none",
        )

        losses = out2.loss.reshape((k, -1)).sum(dim=1)

        # copy the losses over to another vector
        scores = torch.full(
            [n_choices], torch.finfo(torch.float).min, device=losses.device
        )
        scores[topk_indices] = -losses
        return scores.detach()



class PerceiverAttentionLayer(nn.Module):
    def __init__(
            self,
            *,
            dim,
            dim_head=64,
            heads=8
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        self.dim_head = dim_head
        inner_dim = dim_head * heads

        # trainable components of PerceiverAttentionLayer
        self.norm_media = nn.LayerNorm(dim)
        self.norm_latents = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, features, latents):
        """
        Latent vectors are cross-attending to the visual features x.
        :param x:       Tensor (n_batch, n_features, dim)
                        visual features
        :param latents: Tensor (n_batch, n_latents, dim)
                        latent learnt vectors from which the queries are computed.
                        Actually the same, just replicated in n_batch and n_frames dimension.
        :return:        Tensor (n_batch, n_latents, dim)
        """
        assert features.ndim == 3
        assert latents.ndim == 3
        assert features.shape[0] == latents.shape[0]
        assert features.shape[2] == latents.shape[2]

        n_heads = self.heads
        n_batch, n_features, dim = features.shape
        n_queries = latents.shape[1]

        # layer normalization, as usual
        x = self.norm_media(features)
        latents = self.norm_latents(latents)

        # queries
        # compute the queries from the latents, for all attention heads simultaneously.
        q = self.to_q(latents)
        q = rearrange(q, 'b q (h d) -> b h q d', h=n_heads)
        assert q.shape == torch.Size([n_batch, n_heads, n_queries, self.dim_head])

        # keys and values for all attention heads
        # kv_input = torch.cat((x, latents), dim=-2)
        # k, v = self.to_kv(kv_input).chunk(2, dim=-1)
        # latents_ = repeat(latents, 'q d -> b q d', b=n_batch)
        kv_input = torch.cat((x, latents), dim=-2)
        n_features_latents = n_features + n_queries

        # keys, values
        k = self.to_k(kv_input)
        v = self.to_v(kv_input)
        # batch, features, (heads, dim)

        # split so we have an extra dimension for the heads
        # q, k, v = rearrange_many((q, k, v), 'b t n (h d) -> b h t n d', h=h)
        k, v = rearrange_many((k, v), 'b f (h d) -> b h f d', h=n_heads)
        assert v.shape == torch.Size([n_batch, n_heads, n_features_latents, self.dim_head])

        # scale queries?
        q = q * self.scale

        # attention

        # attention scores
        # sim = einsum('... i d, ... j d  -> ... i j', q, k)
        sim = einsum('b h q d, b h f d -> b h q f', q, k)

        # Is this for numerical stability? Does not affect the result of the softmax operation
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        alphas = sim.softmax(dim=-1)

        # out = einsum('... i j, ... j d -> ... i d', alphas, v)
        out = einsum('b h q f, b h f v -> b h q v', alphas, v)

        # out = rearrange(out, 'b h t n d -> b t n (h d)', h=h)
        out = rearrange(out, 'b h q v -> b q (h v)')
        return self.to_out(out)


class PerceiverResampler(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth,
        dim_head=64,
        heads=8,
        num_latents=64,
        num_time_embeds=4,
        ff_mult=4,
        act='gelu'
    ):
        """
        :param dim:             length of the visual features and of the queries (-> thus also length of the keys)
        :param depth:           number of attention layers
        :param dim_head:        inner dimensionality of the q, k, v vectors per attention head
        :param heads:           number of attention heads
        :param num_latents:     number of queries, default 64 as in the flamingo paper
        :param num_time_embeds: TODO what is this? maximum number of frames per video clip? Then 4 seems not enough
        :param ff_mult:         factor for the number of inner neurons in the feedforward layer
        """
        super().__init__()

        self.dim = dim
        self.n_queries = num_latents

        # latents are not the queries themselves, but latent vectors from which the queries are computed.
        # (same dimension as the length of the features
        self.latents = nn.Parameter(torch.randn(num_latents, dim))

        # the time positional embeddings are learnable parameters as well?
        self.time_pos_emb = nn.Parameter(torch.randn(num_time_embeds, 1, dim))

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PerceiverAttentionLayer(dim=dim, dim_head=dim_head, heads=heads),
                FeedForward(dim=dim, mult=ff_mult, act=act)
            ]))

        # layer normalization takes as input the query vector length
        self.norm = nn.LayerNorm(dim)

    def forward(self, x_f):
        """
        TODO does the output dimensionality of visual features does need to match the input dimensionality? (d_visual)
        d_visual must be equal to the dimensionality of the latents, since they are concatenated in the xattn layer.
        :param x_f: Tensor (n_batch, n_features, d_visual) or (n_batch, n_frames, n_features, d_visual)
        :return:    Tensor (n_batch, T, n_queries, d_visual)
        """
        if x_f.ndim == 3:
            # if batch is just a tensor of images, extend frame dimension -> images are like videos with just one frame
            x_f = rearrange(x_f, 'b n d -> b 1 n d')

        assert x_f.ndim == 4

        n_batches = x_f.shape[0]
        n_frames = x_f.shape[1]
        n_visual_features = x_f.shape[2]
        dim = x_f.shape[3]

        assert dim == self.dim

        # add time embeddings
        # for each frame. Times is the number of frames
        # time embedding is added to every visual feature of one frame.
        x_f = x_f + self.time_pos_emb[:n_frames]

        # >> David added
        # flatten frames
        # not sure if it makes a difference, but lucidrains did not flatten the features from the frames.
        # It makes a difference in the output shape, if we are processing video clips.
        x_f = rearrange(x_f, 'b T n d -> b (T n) d')

        # >> David modified
        # copy the latents for every element in the batch.
        # lucidrains did extend the latents to b T q d, however makes no sense if frames are flattened before
        # q = number of queries
        # d = dimension of queries
        x = repeat(self.latents, 'q d -> b q d', b=n_batches)

        for attn, ffw in self.layers:  # type: ignore
            x = x + attn(x_f, x)
            x = x + ffw(x)

        assert x.shape == torch.Size([n_batches, self.n_queries, self.dim])

        norm = self.norm(x)
        return norm


