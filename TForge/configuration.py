from transformers import PretrainedConfig

class TForgeConfig(PretrainedConfig):
    model_type = "Transformer_Forge"

    def __init__(
        self, input_vocab_size: int=10000, output_vocab_size: int=10000, translate_vocab_size: int=10000,
        max_positions: int=1024, num_layers: int=6, num_heads: int=8, embed_dim:int=768, hidden_dim:int=3072,
        soft_align_dim: int = 512, dropout_prob: float=0.1, pad_token_id: int=1, eos_token_id: int=3,
        is_encoder_decoder: bool=True, decoder_start_token_id: int=2, forced_eos_token_id: int=3,
        load_pretrained: bool=False, **kwargs
    ):
        """Constructs TForgeConfig."""
        super().__init__(
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            is_encoder_decoder=is_encoder_decoder,
            decoder_start_token_id=decoder_start_token_id,
            forced_eos_token_id=forced_eos_token_id,
            **kwargs,
        )
        self.input_vocab_size = input_vocab_size
        self.output_vocab_size = output_vocab_size
        self.translate_vocab_size = translate_vocab_size
        self.max_positions = max_positions
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.soft_align_dim = soft_align_dim
        self.dropout_prob = dropout_prob
        self.load_pretrained = load_pretrained
        