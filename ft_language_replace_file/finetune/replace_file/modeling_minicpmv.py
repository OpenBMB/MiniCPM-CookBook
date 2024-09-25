import math
from typing import List, Optional
import json
import torch
import torchvision
from threading import Thread
from copy import deepcopy
from PIL import Image
from torchvision import transforms
from transformers import LlamaTokenizer, LlamaPreTrainedModel, LlamaForCausalLM, AutoModel, PreTrainedTokenizerFast, TextIteratorStreamer
from transformers.models.idefics2.modeling_idefics2 import Idefics2VisionTransformer
import logging
from .configuration_minicpm import MiniCPMVConfig
from .resampler import Resampler


IMAGENET_INCEPTION_MEAN = (0.5, 0.5, 0.5) # timm.data.IMAGENET_INCEPTION_MEAN
IMAGENET_INCEPTION_STD = (0.5, 0.5, 0.5)  # timm.data.IMAGENET_INCEPTION_STD
logger=logging.getLogger('my_logger')
logger.setLevel(logging.DEBUG)
file_handler=logging.FileHandler(filename='/root/ld/ld_project/MiniCPM-V/finetune/output/log/finetune.log')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
class MiniCPMVPreTrainedModel(LlamaPreTrainedModel):
    config_class = MiniCPMVConfig


class MiniCPMV(MiniCPMVPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.llm = LlamaForCausalLM(config)
        self.vpm = self.init_vision_module()
        self.vision_dim = self.vpm.embed_dim
        self.embed_dim = self.llm.config.hidden_size
        self.resampler = self.init_resampler(self.embed_dim, self.vision_dim)
        self.transform = self.init_transform()

    def init_vision_module(self):
        # same as HuggingFaceM4/siglip-so400m-14-980-flash-attn2-navit
        model = Idefics2VisionTransformer(self.config.vision_config)
        if self.config.drop_vision_last_layer:
            model.encoder.layers = model.encoder.layers[:-1]

        setattr(model, 'embed_dim', model.embeddings.embed_dim)
        setattr(model, 'patch_size', model.embeddings.patch_size)

        return model

    def init_resampler(self, embed_dim, vision_dim,):
        return Resampler(
            num_queries=self.config.query_num,
            embed_dim=embed_dim,
            num_heads=embed_dim // 128,
            kv_dim=vision_dim,
            adaptive=True,
        )

    def init_transform(self):
        return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD
                ),
            ]
        )
    
    def get_input_embeddings(self):
        return self.llm.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.llm.embed_tokens = value
        
    def get_vllm_embedding(self, data):
        if 'vision_hidden_states' not in data:
            dtype = self.llm.model.embed_tokens.weight.dtype
            device = self.llm.model.embed_tokens.weight.device
            tgt_sizes = data['tgt_sizes']

            
            pixel_values_list = data['pixel_values']
            vision_hidden_states = []
            all_pixel_values = []
            img_cnt = []
            for pixel_values in pixel_values_list:
                img_cnt.append(len(pixel_values))
                #####
                if pixel_values==[]:
                    continue
                #####
                all_pixel_values.extend([i.flatten(end_dim=1).permute(1, 0) for i in pixel_values])

            # exist image
            if all_pixel_values:
                #logger.info(f'all_pixel_values: {len(all_pixel_values)}')
                tgt_sizes = torch.vstack(tgt_sizes).type(torch.int32)
                #####这里是增加的代码
                #logger.info(f"这里是tgt_sizes: {tgt_sizes}")
                mask = (tgt_sizes == torch.tensor([0, 0], dtype=torch.int32).to(tgt_sizes.device)).all(dim=1)
                no_img_inx=torch.nonzero(mask, as_tuple=False).squeeze(1)
                tgt_sizes = tgt_sizes[~mask]
                #logger.info(f"tgt_sizes:{tgt_sizes}")
                #####
                #####
                #logger.info(f'tgt_sizes: {tgt_sizes.shape}')
                if self.config.batch_vision_input:
                    max_patches = torch.max(tgt_sizes[:, 0] * tgt_sizes[:, 1])

                    all_pixel_values = torch.nn.utils.rnn.pad_sequence(all_pixel_values, batch_first=True,
                                                                       padding_value=0.0)
                    B, L, _ = all_pixel_values.shape
                    all_pixel_values = all_pixel_values.permute(0, 2, 1).reshape(B, 3, -1, L)

                    patch_attn_mask = torch.zeros((B, 1, max_patches), dtype=torch.bool, device=device)
                    for i in range(B):
                        patch_attn_mask[i, :tgt_sizes[i][0] * tgt_sizes[i][1]] = True

                    vision_embedding = self.vpm(all_pixel_values.type(dtype), patch_attention_mask=patch_attn_mask).last_hidden_state
                    vision_embedding = self.resampler(vision_embedding, tgt_sizes)
                else:
                    # get vision_embedding foreach
                    vision_embedding = []
                    for single_tgt_size, single_pixel_values in zip(tgt_sizes, all_pixel_values):
                        single_pixel_values = single_pixel_values.unsqueeze(0)
                        B, L, _ = single_pixel_values.shape
                        single_pixel_values = single_pixel_values.permute(0, 2, 1).reshape(B, 3, -1, L)
                        single_vision_embedding = self.vpm(single_pixel_values.type(dtype)).last_hidden_state
                        single_vision_embedding = self.resampler(single_vision_embedding, single_tgt_size.unsqueeze(0))
                        
                        vision_embedding.append(single_vision_embedding)
                    vision_embedding = torch.vstack(vision_embedding)

                start = 0
                for pixel_values in pixel_values_list:
                    img_cnt = len(pixel_values)
                    if img_cnt > 0:
                        vision_hidden_states.append(vision_embedding[start: start + img_cnt])
                        start += img_cnt
                    else:
                        vision_hidden_states.append([])
            else: # no image
                if self.training:
                    dummy_image = torch.zeros(
                        (1, 3, 224, 224),
                        device=device, dtype=dtype
                    )
                    tgt_sizes = torch.Tensor([[(224 // self.config.patch_size), math.ceil(224 / self.config.patch_size)]]).type(torch.int32)
                    dummy_feature = self.resampler(self.vpm(dummy_image).last_hidden_state, tgt_sizes)
                else:
                    dummy_feature = []
                for _ in range(len(pixel_values_list)):
                    vision_hidden_states.append(dummy_feature)

        else:
            vision_hidden_states = data['vision_hidden_states']

        if hasattr(self.llm.config, 'scale_emb'):
            vllm_embedding = self.llm.model.embed_tokens(data['input_ids']) * self.llm.config.scale_emb
        else:
            vllm_embedding = self.llm.model.embed_tokens(data['input_ids'])

        vision_hidden_states = [i.type(vllm_embedding.dtype) if isinstance(
            i, torch.Tensor) else i for i in vision_hidden_states]

        bs = len(data['input_ids'])
        for i in range(bs):
            cur_vs_hs = vision_hidden_states[i]
            if len(cur_vs_hs) > 0:
                cur_vllm_emb = vllm_embedding[i]
                cur_image_bound = data['image_bound'][i]
                if len(cur_image_bound) > 0:
                    image_indices = torch.stack(
                        [torch.arange(r[0], r[1], dtype=torch.long) for r in cur_image_bound]
                    ).to(vllm_embedding.device)
                    cur_vllm_emb.scatter_(0, image_indices.view(-1, 1).repeat(1, cur_vllm_emb.shape[-1]),
                                          cur_vs_hs.view(-1, cur_vs_hs.shape[-1]))
                elif self.training:
                    cur_vllm_emb += cur_vs_hs[0].mean() * 0

        return vllm_embedding, vision_hidden_states
        
    def forward(self, data, **kwargs):
        """data={
        "input_ids": input_ids,
        "position_ids": position_ids,
        "labels": targets,
        "attention_mask": attention_mask,
        "image_bound": image_bound,
        "tgt_sizes": tgt_sizes,
        "pixel_values": pixel_values,
    }"""
        vllm_embedding, vision_hidden_states = self.get_vllm_embedding(data)
        position_ids = data["position_ids"]
        if position_ids.dtype != torch.int64:
            position_ids = position_ids.long()

        return self.llm(
            input_ids=None,
            position_ids=position_ids,
            inputs_embeds=vllm_embedding,
            **kwargs
        )

    def _convert_to_tensors(
        self, tokenizer, input_ids, max_inp_length: Optional[int] = None
    ):
        if max_inp_length is not None:
            input_ids = input_ids[:max_inp_length]
        input_ids = torch.tensor(input_ids, dtype=torch.int32)

        image_start_tokens = torch.where(input_ids == tokenizer.im_start_id)[0]
        # 跳过 im_start
        image_start_tokens += 1
        image_end_tokens = torch.where(input_ids == tokenizer.im_end_id)[0]
        valid_image_nums = max(len(image_start_tokens), len(image_end_tokens))
        image_bound = torch.hstack(
            [
                image_start_tokens[:valid_image_nums].unsqueeze(-1),
                image_end_tokens[:valid_image_nums].unsqueeze(-1),
            ]
        )

        model_input = {}
        model_input["input_ids"] = input_ids.unsqueeze(0).to(self.device)
        model_input["image_bound"] = image_bound

        return model_input

    def _process_list(
        self, tokenizer, input_id_list, max_inp_length: Optional[int] = None
    ):
        pad_keys = ["input_ids"]
        input_tensors = []
        for input_ids in input_id_list:
            input_tensors.append(
                self._convert_to_tensors(tokenizer, input_ids, max_inp_length)
            )
        padded = {}
        for key in pad_keys:
            padded[key] = pad(input_tensors, key, padding_side="left").to(self.device)
        padded["image_bound"] = [i["image_bound"] for i in input_tensors]
        return padded

    def _decode(self, inputs_embeds, tokenizer, **kwargs):
        terminators = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        output = self.llm.generate(
            inputs_embeds=inputs_embeds,
            pad_token_id=0,
            eos_token_id=terminators,
            **kwargs
        )
        return self._decode_text(output, tokenizer)
    
    def _decode_stream(self, inputs_embeds, tokenizer, **kwargs):
        terminators = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        streamer = TextIteratorStreamer(tokenizer=tokenizer)
        generation_kwargs = {
            'inputs_embeds': inputs_embeds,
            'pad_token_id': 0,
            'eos_token_id': terminators,
            'streamer': streamer
        }
        generation_kwargs.update(kwargs)

        thread = Thread(target=self.llm.generate, kwargs=generation_kwargs)
        thread.start()
    
        return streamer

    def _decode_text(self, result_ids, tokenizer):
        result_text = []
        for result in result_ids:
            result = result[result != 0]
            if result[0] == tokenizer.bos_id:
                result = result[1:]
            if result[-1] == tokenizer.eos_id or result[-1] == tokenizer.eot_id:
                result = result[:-1]
            result_text.append(tokenizer.decode(result).strip())
        return result_text

    def slice_image(self, image):
        return slice_image(
            image,
            self.config.slice_config.max_slice_nums,
            self.config.slice_config.scale_resolution,
            self.config.slice_config.patch_size,
        )

    def get_slice_image_placeholder(self, image, tokenizer):
        image_placeholder = (
            tokenizer.im_start
            + tokenizer.unk_token * self.config.query_num
            + tokenizer.im_end
        )

        slice_images = []

        source_image, patches, best_grid = slice_image(
            image,
            self.config.slice_config.max_slice_nums,
            self.config.slice_config.scale_resolution,
            self.config.slice_config.patch_size,
        )

        slice_images.append(source_image)
        final_placeholder = image_placeholder

        if len(patches) > 0:
            for i in range(len(patches)):
                for j in range(len(patches[0])):
                    slice_images.append(patches[i][j])

            final_placeholder += get_grid_placeholder(
                tokenizer, best_grid, self.config.query_num
            )

        return slice_images, final_placeholder

    def reshape_by_patch(self, image_tensor):
        """
        :param image_tensor: shape [3, H, W]
        :param patch_size:
        :return: [3, patch_size, HW/patch_size]
        """
        patch_size = self.config.patch_size
        patches = torch.nn.functional.unfold(
            image_tensor,
            (patch_size, patch_size),
            stride=(patch_size, patch_size)
        )

        patches = patches.reshape(image_tensor.size(0), patch_size, patch_size, -1)
        patches = patches.permute(0, 1, 3, 2).reshape(image_tensor.size(0), patch_size, -1)
        return patches

    def generate(
        self,
        input_id_list=None,
        img_list=None,
        tgt_sizes=None,
        tokenizer=None,
        max_inp_length: Optional[int] = None,
        vision_hidden_states=None,
        return_vision_hidden_states=False,
        stream=False,
        **kwargs
    ):

        assert input_id_list is not None
        bs = len(input_id_list)
        if img_list == None:
            img_list = [[] for i in range(bs)]
        assert bs == len(img_list)

        model_inputs = self._process_list(tokenizer, input_id_list, max_inp_length)

        if vision_hidden_states is None:
            pixel_values = []
            for i in range(bs):
                img_inps = []
                for img in img_list[i]:
                    img_inps.append(img.to(self.device))
                if img_inps:
                    pixel_values.append(img_inps)
                else:
                    pixel_values.append([])
            model_inputs["pixel_values"] = pixel_values
            model_inputs['tgt_sizes'] = tgt_sizes
        else:
            model_inputs["vision_hidden_states"] = vision_hidden_states

        with torch.inference_mode():
            (
                model_inputs["inputs_embeds"],
                vision_hidden_states,
            ) = self.get_vllm_embedding(model_inputs)

            if stream:
                result = self._decode_stream(model_inputs["inputs_embeds"], tokenizer, **kwargs)
            else:
                result = self._decode(model_inputs["inputs_embeds"], tokenizer, **kwargs)

        if return_vision_hidden_states:
            return result, vision_hidden_states

        return result

    def chat(
        self,
        image,
        msgs,
        tokenizer,
        vision_hidden_states=None,
        max_new_tokens=1024,
        sampling=True,
        max_inp_length=2048,
        system_prompt='',
        stream=False,
        **kwargs
    ):
        if isinstance(msgs, str):
            msgs = json.loads(msgs)

        copy_msgs = deepcopy(msgs)
        assert len(copy_msgs) > 0, 'msgs is empty'
        assert sampling or not stream, 'if use stream mode, make sure sampling=True'

        if image is not None and isinstance(copy_msgs[0]['content'], str):
            copy_msgs[0]['content'] = [image, copy_msgs[0]['content']]

        images = []
        tgt_sizes = []
        for i, msg in enumerate(copy_msgs):
            role = msg["role"]
            content = msg["content"]
            assert role in ["user", "assistant"]
            if i == 0:
                assert role == "user", "The role of first msg should be user"
            if isinstance(content, str):
                content = [content]

            cur_msgs = []
            for c in content:
                if isinstance(c, Image.Image):
                    image = c
                    if self.config.slice_mode:
                        slice_images, image_placeholder = self.get_slice_image_placeholder(
                            image, tokenizer
                        )
                        cur_msgs.append(image_placeholder)
                        for slice_image in slice_images:
                            slice_image = self.transform(slice_image)
                            H, W = slice_image.shape[1:]
                            images.append(self.reshape_by_patch(slice_image))
                            tgt_sizes.append(torch.Tensor([H // self.config.patch_size, W // self.config.patch_size]).type(torch.int32))
                    else:
                        images.append(self.transform(image))
                        cur_msgs.append(
                            tokenizer.im_start
                            + tokenizer.unk_token * self.config.query_num
                            + tokenizer.im_end
                        )
                elif isinstance(c, str):
                    cur_msgs.append(c)


            msg['content'] = '\n'.join(cur_msgs)
        if tgt_sizes:
            tgt_sizes = torch.vstack(tgt_sizes)

        if system_prompt:
            sys_msg = {'role': 'system', 'content': system_prompt}
            copy_msgs = [sys_msg] + copy_msgs

        input_ids = tokenizer.apply_chat_template(copy_msgs, tokenize=True, add_generation_prompt=False)

        if sampling:
            generation_config = {
                "top_p": 0.8,
                "top_k": 100,
                "temperature": 0.7,
                "do_sample": True,
                "repetition_penalty": 1.05
            }
        else:
            generation_config = {
                "num_beams": 3,
                "repetition_penalty": 1.2,
            }

        generation_config.update(
            (k, kwargs[k]) for k in generation_config.keys() & kwargs.keys()
        )

        with torch.inference_mode():
            res, vision_hidden_states = self.generate(
                input_id_list=[input_ids],
                max_inp_length=max_inp_length,
                img_list=[images],
                tgt_sizes=[tgt_sizes],
                tokenizer=tokenizer,
                max_new_tokens=max_new_tokens,
                vision_hidden_states=vision_hidden_states,
                return_vision_hidden_states=True,
                stream=stream,
                **generation_config
            )

        if stream:
            def stream_gen():
                for text in res:
                    text = text.replace(tokenizer.eot_token, '').replace(tokenizer.eos_token, '')
                    yield text
            return stream_gen()

        else:
            answer = res[0]
            return answer


class PreTrainedTokenizerFastWrapper(PreTrainedTokenizerFast):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.eot_token = "<|eot_id|>"
        self.im_start = "<image>"
        self.im_end = "</image>"
        self.ref_start = "<ref>"
        self.ref_end = "</ref>"
        self.box_start = "<box>"
        self.box_end = "</box>"
        self.quad_start = "<quad>"
        self.quad_end = "</quad>"
        self.slice_start = "<slice>"
        self.slice_end = "</slice>"

    @property
    def eos_id(self):
        return self.eos_token_id

    @property
    def bos_id(self):
        return self.bos_token_id

    @property
    def unk_id(self):
        return self.unk_token_id

    @property
    def eot_id(self):
        return self.convert_tokens_to_ids(self.eot_token)

    @property
    def im_start_id(self):
        return self.convert_tokens_to_ids(self.im_start)

    @property
    def im_end_id(self):
        return self.convert_tokens_to_ids(self.im_end)

    @staticmethod
    def escape(text: str) -> str:
        return text

    @staticmethod
    def unescape(text: str) -> str:
        return text


def pad(orig_items, key, max_length=None, padding_value=0, padding_side="left"):
    items = []
    if isinstance(orig_items[0][key], list):
        assert isinstance(orig_items[0][key][0], torch.Tensor)
        for it in orig_items:
            for tr in it[key]:
                items.append({key: tr})
    else:
        assert isinstance(orig_items[0][key], torch.Tensor)
        items = orig_items

    batch_size = len(items)
    shape = items[0][key].shape
    dim = len(shape)
    assert dim <= 3
    if max_length is None:
        max_length = 0
    max_length = max(max_length, max(item[key].shape[-1] for item in items))
    min_length = min(item[key].shape[-1] for item in items)
    dtype = items[0][key].dtype

    if dim == 1:
        return torch.cat([item[key] for item in items], dim=0)
    elif dim == 2:
        if max_length == min_length:
            return torch.cat([item[key] for item in items], dim=0)
        tensor = torch.zeros((batch_size, max_length), dtype=dtype) + padding_value
    else:
        tensor = (
            torch.zeros((batch_size, max_length, shape[-1]), dtype=dtype)
            + padding_value
        )

    for i, item in enumerate(items):
        if dim == 2:
            if padding_side == "left":
                tensor[i, -len(item[key][0]) :] = item[key][0].clone()
            else:
                tensor[i, : len(item[key][0])] = item[key][0].clone()
        elif dim == 3:
            if padding_side == "left":
                tensor[i, -len(item[key][0]) :, :] = item[key][0].clone()
            else:
                tensor[i, : len(item[key][0]), :] = item[key][0].clone()

    return tensor


def slice_image(
    image, max_slice_nums=9, scale_resolution=448, patch_size=14, never_split=False
):
    original_size = image.size
    original_width, original_height = original_size
    log_ratio = math.log(original_width / original_height)
    ratio = original_width * original_height / (scale_resolution * scale_resolution)
    multiple = min(math.ceil(ratio), max_slice_nums)

    source_image = None
    best_grid = None
    patches = []

    if multiple <= 1 or never_split:
        # dont need to slice, upsample
        best_size = find_best_resize(
            original_size, scale_resolution, patch_size, allow_upscale=True
        )
        source_image = image.resize(best_size, Image.Resampling.BICUBIC)
    else:
        candidate_split_grids_nums = []
        for i in [multiple - 1, multiple, multiple + 1]:
            if i == 1 or i > max_slice_nums:
                continue
            candidate_split_grids_nums.append(i)

        # source image, down-sampling and ensure divided by patch_size
        best_resize = find_best_resize(original_size, scale_resolution, patch_size)
        source_image = image.copy().resize(best_resize, Image.Resampling.BICUBIC)
        candidate_grids = []

        # find best grid
        for split_grids_nums in candidate_split_grids_nums:
            m = 1
            while m <= split_grids_nums:
                if split_grids_nums % m == 0:
                    candidate_grids.append([m, split_grids_nums // m])
                m += 1

        best_grid = [1, 1]
        min_error = float("inf")
        for grid in candidate_grids:
            error = abs(log_ratio - math.log(grid[0] / grid[1]))
            if error < min_error:
                best_grid = grid
                min_error = error

        refine_size = get_refine_size(
            original_size, best_grid, scale_resolution, patch_size, allow_upscale=True
        )

        refine_image = image.resize(refine_size, Image.Resampling.BICUBIC)
        patches = split_to_patches(refine_image, best_grid)

    return source_image, patches, best_grid


def ensure_divide(length, patch_size):
    return max(round(length / patch_size) * patch_size, patch_size)


def find_best_resize(original_size, scale_resolution, patch_size, allow_upscale=False):
    width, height = original_size
    if (width * height > scale_resolution * scale_resolution) or allow_upscale:
        r = width / height
        height = int(scale_resolution / math.sqrt(r))
        width = int(height * r)
    best_width = ensure_divide(width, patch_size)
    best_height = ensure_divide(height, patch_size)
    return (best_width, best_height)


def get_refine_size(
    original_size, grid, scale_resolution, patch_size, allow_upscale=False
):
    width, height = original_size
    grid_x, grid_y = grid

    refine_width = ensure_divide(width, grid_x)
    refine_height = ensure_divide(height, grid_y)

    grid_width = refine_width / grid_x
    grid_height = refine_height / grid_y

    best_grid_size = find_best_resize(
        (grid_width, grid_height),
        scale_resolution,
        patch_size,
        allow_upscale=allow_upscale,
    )

    refine_size = (best_grid_size[0] * grid_x, best_grid_size[1] * grid_y)

    return refine_size


def split_to_patches(image, grid):
    patches = []
    width, height = image.size
    grid_x = int(width / grid[0])
    grid_y = int(height / grid[1])

    for i in range(0, height, grid_y):
        images = []
        for j in range(0, width, grid_x):
            box = (j, i, j + grid_x, i + grid_y)
            patch = image.crop(box)
            images.append(patch)
        patches.append(images)

    return patches


def get_grid_placeholder(tokenizer, grid, query_num):
    image_placeholder = (
        tokenizer.im_start + tokenizer.unk_token * query_num + tokenizer.im_end
    )

    cols = grid[0]
    rows = grid[1]
    slices = []
    for i in range(rows):
        lines = []
        for j in range(cols):
            lines.append(image_placeholder)
        slices.append("".join(lines))
    slice_placeholder = tokenizer.slice_start + "\n".join(slices) + tokenizer.slice_end
    return slice_placeholder