import torch
from diffusers.pipelines.stable_diffusion.convert_from_ckpt import convert_ldm_unet_checkpoint, renew_resnet_paths, assign_to_checkpoint, renew_attention_paths, shave_segments

def convert_attention(checkpoint, new_checkpoint, old_prefix, new_prefix, attention_dim=None):
    weight_q, weight_k, weight_v = checkpoint[f"{old_prefix}.qkv.weight"].chunk(3, dim=0)
    bias_q, bias_k, bias_v = checkpoint[f"{old_prefix}.qkv.bias"].chunk(3, dim=0)

    new_checkpoint[f"{new_prefix}.group_norm.weight"] = checkpoint[f"{old_prefix}.norm.weight"]
    new_checkpoint[f"{new_prefix}.group_norm.bias"] = checkpoint[f"{old_prefix}.norm.bias"]

    new_checkpoint[f"{new_prefix}.to_q.weight"] = weight_q.squeeze(-1).squeeze(-1)
    new_checkpoint[f"{new_prefix}.to_q.bias"] = bias_q.squeeze(-1).squeeze(-1)
    new_checkpoint[f"{new_prefix}.to_k.weight"] = weight_k.squeeze(-1).squeeze(-1)
    new_checkpoint[f"{new_prefix}.to_k.bias"] = bias_k.squeeze(-1).squeeze(-1)
    new_checkpoint[f"{new_prefix}.to_v.weight"] = weight_v.squeeze(-1).squeeze(-1)
    new_checkpoint[f"{new_prefix}.to_v.bias"] = bias_v.squeeze(-1).squeeze(-1)

    new_checkpoint[f"{new_prefix}.to_out.0.weight"] = (
        checkpoint[f"{old_prefix}.proj_out.weight"].squeeze(-1).squeeze(-1)
    )
    new_checkpoint[f"{new_prefix}.to_out.0.bias"] = checkpoint[f"{old_prefix}.proj_out.bias"].squeeze(-1).squeeze(-1)

    return new_checkpoint

def convert_struct(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location='cuda:0')
    ckpt = ckpt['state_dict']

    unet_state_dict = {
    }

    new_checkpoint = {
    }

    config = {
        "layers_per_block": 2,
    }

    for key in ckpt.keys():
        if "structcond_stage_model." in key:
            unet_state_dict[key.replace("structcond_stage_model.", "")] = ckpt[key]

    new_checkpoint["conv_in.weight"] = unet_state_dict["input_blocks.0.0.weight"]
    new_checkpoint["conv_in.bias"] = unet_state_dict["input_blocks.0.0.bias"]

    new_checkpoint["time_embedding.linear_1.weight"] = unet_state_dict["time_embed.0.weight"]
    new_checkpoint["time_embedding.linear_1.bias"] = unet_state_dict["time_embed.0.bias"]
    new_checkpoint["time_embedding.linear_2.weight"] = unet_state_dict["time_embed.2.weight"]
    new_checkpoint["time_embedding.linear_2.bias"] = unet_state_dict["time_embed.2.bias"]

    # Retrieves the keys for the input blocks only
    num_input_blocks = len({".".join(layer.split(".")[:2]) for layer in unet_state_dict if "input_blocks" in layer})
    input_blocks = {
        layer_id: [key for key in unet_state_dict if f"input_blocks.{layer_id}" in key]
        for layer_id in range(num_input_blocks)
    }

    # Retrieves the keys for the middle blocks only
    num_middle_blocks = len({".".join(layer.split(".")[:2]) for layer in unet_state_dict if "middle_block" in layer})
    middle_blocks = {
        layer_id: [key for key in unet_state_dict if f"middle_block.{layer_id}" in key]
        for layer_id in range(num_middle_blocks)
    }

    num_output_blocks = len({".".join(layer.split(".")[:2]) for layer in unet_state_dict if "fea_tran" in layer})
    output_blocks = {
        layer_id: [key for key in unet_state_dict if f"fea_tran.{layer_id}" in key]
        for layer_id in range(num_output_blocks)
    }

    for i in range(1, num_input_blocks):
        block_id = (i - 1) // (config["layers_per_block"] + 1)
        layer_in_block_id = (i - 1) % (config["layers_per_block"] + 1)

        resnets = [
            key for key in input_blocks[i] if f"input_blocks.{i}.0" in key and f"input_blocks.{i}.0.op" not in key
        ]
        attentions = [key for key in input_blocks[i] if f"input_blocks.{i}.1" in key]

        if f"input_blocks.{i}.0.op.weight" in unet_state_dict:
            new_checkpoint[f"down_blocks.{block_id}.downsamplers.0.conv.weight"] = unet_state_dict.pop(
                f"input_blocks.{i}.0.op.weight"
            )
            new_checkpoint[f"down_blocks.{block_id}.downsamplers.0.conv.bias"] = unet_state_dict.pop(
                f"input_blocks.{i}.0.op.bias"
            )

        paths = renew_resnet_paths(resnets)
        meta_path = {"old": f"input_blocks.{i}.0", "new": f"down_blocks.{block_id}.resnets.{layer_in_block_id}"}
        assign_to_checkpoint(
            paths, new_checkpoint, unet_state_dict, additional_replacements=[meta_path], config=config
        )

        if len(attentions):

            new_checkpoint = convert_attention(unet_state_dict, new_checkpoint, f"input_blocks.{i}.1", f"down_blocks.{block_id}.attentions.{layer_in_block_id}")
            
            # paths = renew_attention_paths(attentions)
            # meta_path = {"old": f"input_blocks.{i}.1", "new": f"down_blocks.{block_id}.attentions.{layer_in_block_id}"}
            # assign_to_checkpoint(
            #     paths, new_checkpoint, unet_state_dict, additional_replacements=[meta_path], config=config
            # )

    resnet_0 = middle_blocks[0]
    attentions = middle_blocks[1]
    resnet_1 = middle_blocks[2]

    resnet_0_paths = renew_resnet_paths(resnet_0)
    assign_to_checkpoint(resnet_0_paths, new_checkpoint, unet_state_dict, config=config)

    new_checkpoint = convert_attention(unet_state_dict, new_checkpoint, f"middle_block.1", f"mid_block.attentions.0")

    resnet_1_paths = renew_resnet_paths(resnet_1)
    assign_to_checkpoint(resnet_1_paths, new_checkpoint, unet_state_dict, config=config)

    resnet_0 = output_blocks[0]
    resnet_1 = output_blocks[1]
    resnet_2 = output_blocks[2]
    resnet_3 = output_blocks[3]

    resnet_0_paths = renew_resnet_paths(resnet_0)
    assign_to_checkpoint(resnet_0_paths, new_checkpoint, unet_state_dict, config=config)

    resnet_1_paths = renew_resnet_paths(resnet_1)
    assign_to_checkpoint(resnet_1_paths, new_checkpoint, unet_state_dict, config=config)

    resnet_2_paths = renew_resnet_paths(resnet_2)
    assign_to_checkpoint(resnet_2_paths, new_checkpoint, unet_state_dict, config=config)

    resnet_3_paths = renew_resnet_paths(resnet_3)
    assign_to_checkpoint(resnet_3_paths, new_checkpoint, unet_state_dict, config=config)

    return new_checkpoint

def convert_unet(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location='cuda:0')
    ckpt = ckpt['state_dict']

    unet_state_dict = {
    }

    new_checkpoint = {
    }

    for key in ckpt.keys():
        if "model.diffusion_model." in key:
            unet_state_dict[key.replace("model.diffusion_model.", "")] = ckpt[key]

    config = {
        "layers_per_block": 2,
    }

    new_checkpoint["conv_in.weight"] = unet_state_dict["input_blocks.0.0.weight"]
    new_checkpoint["conv_in.bias"] = unet_state_dict["input_blocks.0.0.bias"]

    new_checkpoint["time_embedding.linear_1.weight"] = unet_state_dict["time_embed.0.weight"]
    new_checkpoint["time_embedding.linear_1.bias"] = unet_state_dict["time_embed.0.bias"]
    new_checkpoint["time_embedding.linear_2.weight"] = unet_state_dict["time_embed.2.weight"]
    new_checkpoint["time_embedding.linear_2.bias"] = unet_state_dict["time_embed.2.bias"]

    # Retrieves the keys for the input blocks only
    num_input_blocks = len({".".join(layer.split(".")[:2]) for layer in unet_state_dict if "input_blocks" in layer})
    input_blocks = {
        layer_id: [key for key in unet_state_dict if f"input_blocks.{layer_id}" in key]
        for layer_id in range(num_input_blocks)
    }

    # Retrieves the keys for the middle blocks only
    num_middle_blocks = len({".".join(layer.split(".")[:2]) for layer in unet_state_dict if "middle_block" in layer})
    middle_blocks = {
        layer_id: [key for key in unet_state_dict if f"middle_block.{layer_id}" in key]
        for layer_id in range(num_middle_blocks)
    }

    # Retrieves the keys for the output blocks only
    num_output_blocks = len({".".join(layer.split(".")[:2]) for layer in unet_state_dict if "output_blocks" in layer})
    output_blocks = {
        layer_id: [key for key in unet_state_dict if f"output_blocks.{layer_id}" in key]
        for layer_id in range(num_output_blocks)
    }

    for i in range(1, num_input_blocks):
        block_id = (i - 1) // (config["layers_per_block"] + 1)
        layer_in_block_id = (i - 1) % (config["layers_per_block"] + 1)

        resnets = [
            key for key in input_blocks[i] if f"input_blocks.{i}.0" in key and f"input_blocks.{i}.0.op" not in key
        ]
        attentions = [key for key in input_blocks[i] if f"input_blocks.{i}.1" in key]

        if f"input_blocks.{i}.0.op.weight" in unet_state_dict:
            new_checkpoint[f"down_blocks.{block_id}.downsamplers.0.conv.weight"] = unet_state_dict.pop(
                f"input_blocks.{i}.0.op.weight"
            )
            new_checkpoint[f"down_blocks.{block_id}.downsamplers.0.conv.bias"] = unet_state_dict.pop(
                f"input_blocks.{i}.0.op.bias"
            )

        paths = renew_resnet_paths(resnets)
        meta_path = {"old": f"input_blocks.{i}.0", "new": f"down_blocks.{block_id}.resnets.{layer_in_block_id}"}
        assign_to_checkpoint(
            paths, new_checkpoint, unet_state_dict, additional_replacements=[meta_path], config=config
        )

        if len(attentions):
            paths = renew_attention_paths(attentions)

            meta_path = {"old": f"input_blocks.{i}.1", "new": f"down_blocks.{block_id}.attentions.{layer_in_block_id}"}
            assign_to_checkpoint(
                paths, new_checkpoint, unet_state_dict, additional_replacements=[meta_path], config=config
            )

    resnet_0 = middle_blocks[0]
    attentions = middle_blocks[1]
    resnet_1 = middle_blocks[2]

    resnet_0_paths = renew_resnet_paths(resnet_0)
    assign_to_checkpoint(resnet_0_paths, new_checkpoint, unet_state_dict, config=config)

    resnet_1_paths = renew_resnet_paths(resnet_1)
    assign_to_checkpoint(resnet_1_paths, new_checkpoint, unet_state_dict, config=config)

    attentions_paths = renew_attention_paths(attentions)
    meta_path = {"old": "middle_block.1", "new": "mid_block.attentions.0"}
    assign_to_checkpoint(
        attentions_paths, new_checkpoint, unet_state_dict, additional_replacements=[meta_path], config=config
    )

    for i in range(num_output_blocks):
        block_id = i // (config["layers_per_block"] + 1)
        layer_in_block_id = i % (config["layers_per_block"] + 1)
        output_block_layers = [shave_segments(name, 2) for name in output_blocks[i]]
        output_block_list = {}

        for layer in output_block_layers:
            layer_id, layer_name = layer.split(".")[0], shave_segments(layer, 1)
            if layer_id in output_block_list:
                output_block_list[layer_id].append(layer_name)
            else:
                output_block_list[layer_id] = [layer_name]

        if len(output_block_list) > 1:

            resnets = [key for key in output_blocks[i] if f"output_blocks.{i}.0" in key]
            attentions = [key for key in output_blocks[i] if f"output_blocks.{i}.1" in key]

            resnet_0_paths = renew_resnet_paths(resnets)
            paths = renew_resnet_paths(resnets)

            meta_path = {"old": f"output_blocks.{i}.0", "new": f"up_blocks.{block_id}.resnets.{layer_in_block_id}"}
            assign_to_checkpoint(
                paths, new_checkpoint, unet_state_dict, additional_replacements=[meta_path], config=config
            )

            output_block_list = {k: sorted(v) for k, v in sorted(output_block_list.items())}
            if ["conv.bias", "conv.weight"] in output_block_list.values():
                index = list(output_block_list.values()).index(["conv.bias", "conv.weight"])
                new_checkpoint[f"up_blocks.{block_id}.upsamplers.0.conv.weight"] = unet_state_dict[
                    f"output_blocks.{i}.{index}.conv.weight"
                ]
                new_checkpoint[f"up_blocks.{block_id}.upsamplers.0.conv.bias"] = unet_state_dict[
                    f"output_blocks.{i}.{index}.conv.bias"
                ]

                # Clear attentions as they have been attributed above.
                if len(attentions) == 2:
                    attentions = []

            if len(attentions):
                paths = renew_attention_paths(attentions)
                meta_path = {
                    "old": f"output_blocks.{i}.1",
                    "new": f"up_blocks.{block_id}.attentions.{layer_in_block_id}",
                }
                assign_to_checkpoint(
                    paths, new_checkpoint, unet_state_dict, additional_replacements=[meta_path], config=config
                )
        else:
            resnet_0_paths = renew_resnet_paths(output_block_layers, n_shave_prefix_segments=1)
            for path in resnet_0_paths:
                old_path = ".".join(["output_blocks", str(i), path["old"]])
                new_path = ".".join(["up_blocks", str(block_id), "resnets", str(layer_in_block_id), path["new"]])

                new_checkpoint[new_path] = unet_state_dict[old_path]

    new_checkpoint["conv_norm_out.weight"] = unet_state_dict["out.0.weight"]
    new_checkpoint["conv_norm_out.bias"] = unet_state_dict["out.0.bias"]
    new_checkpoint["conv_out.weight"] = unet_state_dict["out.2.weight"]
    new_checkpoint["conv_out.bias"] = unet_state_dict["out.2.bias"]

    return new_checkpoint

if __name__ == "__main__":
    ckpt_path = "/NEW_EDS/JJ_Group/xutd/StableSR/logs/2024-08-02T13-58-31_sdsr_reimpl_debug/checkpoints/last.ckpt"
    unet_ckpt = convert_unet(ckpt_path)
    struct_ckpt = convert_struct(ckpt_path)
    torch.save({
        "struct": struct_ckpt,
        "unet": unet_ckpt,
    },
        "/NEW_EDS/JJ_Group/xutd/diffusion-inversion/models/sdsr_reimpl/sdsr.ckpt"
    )