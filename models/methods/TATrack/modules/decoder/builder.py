def build_decoder(config: dict, drop_path_allocator,
                  dim, num_heads, mlp_ratio, qkv_bias, drop_rate, attn_drop_rate,
                  z_shape, x_shape):


    if config['transformer_type'] == 'concatenation_feature_fusion':
        from .concatenated_fusion import build_feature_map_generation_decoder
        return build_feature_map_generation_decoder(config, drop_path_allocator,
                                                    dim, num_heads, mlp_ratio, qkv_bias, drop_rate, attn_drop_rate,
                                                    z_shape, x_shape)
    elif config['transformer_type'] == 'target_query_decoder':
        from .target_query_decoder import build_target_query_decoder
        return build_target_query_decoder(config, drop_path_allocator,
                                          dim, num_heads, mlp_ratio, qkv_bias, drop_rate, attn_drop_rate,
                                          z_shape, x_shape)
    else:
        raise NotImplementedError(config['transformer_type'])
