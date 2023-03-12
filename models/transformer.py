import torch
from torch import nn
import numpy as np


class Transformer(nn.Module):
    def __init__(
        self,
        x_vocab_size,
        y_vocab_size,
        embedding_dim,
        n_head,
        head_dim,
        feed_forward_dim,
        n_encoder,
        n_decoder,
        pad_idx,
        negative_inf=-1e9,
    ):
        super().__init__()

        self.x_vocab_size = x_vocab_size
        self.y_vocab_size = y_vocab_size
        self.pad_idx = pad_idx
        self.negative_inf = negative_inf
        self.embedding_dim = embedding_dim
        self.x_embedding = nn.Embedding(x_vocab_size, embedding_dim)
        self.y_embedding = nn.Embedding(y_vocab_size, embedding_dim)
        self.encoder = Encoder(embedding_dim, n_head, head_dim, feed_forward_dim, n_encoder)
        self.decoder = Decoder(embedding_dim, n_head, head_dim, feed_forward_dim, n_decoder)
        # (batch, seq_len, embed_dim) -> (batch, seq_len, y_vocab_size)
        self.decode_linear = nn.Linear(embedding_dim, y_vocab_size)

    def forward(self, x, y):
        """transformer의 forward.

        Args:
            x (tensor): (batch, x_seq_len)
            y (tensor): (batch, y_seq_len)

        Returns:
            _type_: _description_
        """
        ## encoder embedding.
        # embedding.
        x_embeded = self.x_embedding(x)  # (batch, seq_len, embed_dim)

        # positional encoding. # TODO: pad_mask 추가?
        pos_encoding_x = self.postional_encoding(x, self.embedding_dim)
        inputs_x = x_embeded + torch.FloatTensor(pos_encoding_x)  # (batch, seq_len, embed_dim)

        ## decoder embedding.
        # embedding.
        y_embeded = self.y_embedding(y)  # (batch, seq_len, embed_dim)

        # positional encoding. # TODO: pad_mask 추가?
        pos_encoding_y = self.postional_encoding(y, self.embedding_dim)
        inputs_y = y_embeded + torch.FloatTensor(pos_encoding_y)  # (batch, seq_len, embed_dim)

        ## generate masks.
        # generate encoder mask.
        encoder_pad_mask = self.generate_square_pad_mask(x, x, self.pad_idx)  # (batch, x_seq_len, x_seq_len)

        # generate decoder mask.
        decoder_pad_mask = self.generate_square_pad_mask(y, y, self.pad_idx)  # (batch, y_seq_len, y_seq_len)
        decoder_subsequent_mask = self.generate_square_subsequent_mask(y.shape[1])  # (y_seq_len, y_seq_len)
        decoder_mask = decoder_pad_mask + decoder_subsequent_mask  # (batch, y_seq_len, y_seq_len)

        # generate encoder-decoder mask.
        encoder_decoder_pad_mask = self.generate_square_pad_mask(
            y, x, self.pad_idx
        )  # decoder query to encoder key. (batch, y_seq_len, x_seq_len)

        ## foward encoder-decoder modules.
        # encoder module.
        encoder_output = self.encoder(inputs_x, encoder_pad_mask)  # (batch, seq_len, embed_dim)

        # decoder module.
        decoder_output = self.decoder(inputs_y, encoder_output, decoder_mask, encoder_decoder_pad_mask)

        ## output process.
        decoder_output_prob = self.decode_linear(decoder_output)
        decoder_output_prob = torch.softmax(decoder_output_prob, dim=-1)
        output = torch.argmax(decoder_output_prob, dim=-1)

        return output

    def postional_encoding(self, input, embedding_dim):
        """
        positional 인코딩.
        (seqence length, embedding dimension) 크기의 positional encoding matrix가 생성.

        input의 shape을 바탕으로 batch를 맞춰주고,
        padding token에 해당한는 값은 positional_encoding을 수행하지 않음.
        """
        batch_size, seq_len = input.shape
        encoding_matrix = np.zeros((seq_len, embedding_dim))

        for seq_pos in range(encoding_matrix.shape[0]):
            for emb_pos in range(encoding_matrix.shape[1]):
                # 문장에서의 위치와 임베딩 dim 에서의 위치에 따른 angle.
                angle = seq_pos / np.power(10000, 2 * emb_pos / embedding_dim)
                if emb_pos % 2 == 0:
                    # 짝수는 sin.
                    sinusoid = np.sin
                else:
                    # 홀수는 cos.
                    sinusoid = np.cos

                encoding_matrix[seq_pos][emb_pos] = sinusoid(angle)

        ## draw positional encoding matrix.
        # plt.pcolormesh(encoding_matrix)
        # plt.xlabel('embedding_position')
        # plt.ylabel('seqence position')
        # plt.savefig('test.png')

        # batch 크기 만큼 복사.
        encoding_matrix = np.tile(encoding_matrix, (batch_size, 1, 1))

        return encoding_matrix

    def generate_square_subsequent_mask(self, size: int) -> torch.Tensor:
        """이후 단어를 미리 인식하지 못하도록 mask.
        +------+-----+-------+------+
        |      | i   | like  | you  |
        +------+-----+-------+------+
        | i    |  0  | -inf  | -inf |
        +------+-----+-------+------+
        | like |  0  |   0   | -inf |
        +------+-----+-------+------+
        | you  |  0  |   0   |  0   |
        +------+-----+-------+------+
        """
        return torch.triu(torch.ones(size, size) * self.negative_inf, diagonal=1)

    def generate_square_pad_mask(self, query, key, pad_idx) -> torch.Tensor:
        """padding에 해당하는 부분은 계산하지 않도록 mask.

        +-------+------+------+------+-------+-------+
        |       | i    | like | you  | <pad> | <pad> |
        +-------+------+------+------+-------+-------+
        | i     | 0    | 0    | 0    | -inf  | -inf  |
        +-------+------+------+------+-------+-------+
        | like  | 0    | 0    | 0    | -inf  | -inf  |
        +-------+------+------+------+-------+-------+
        | you   | 0    | 0    | 0    | -inf  | -inf  |
        +-------+------+------+------+-------+-------+
        | <pad> | -inf | -inf | -inf | -inf  | -inf  |
        +-------+------+------+------+-------+-------+
        | <pad> | -inf | -inf | -inf | -inf  | -inf  |
        +-------+------+------+------+-------+-------+

        Args:
            query (_type_): (batch, seq_len)
            key (_type_): (batch, seq_len)

        Returns:
            torch.Tensor: (batch, seq_len, seq_len)
        """
        # padding_mask

        pad_mask = torch.zeros(query.shape[0], query.shape[1], key.shape[1])
        pad_mask[torch.where(query == pad_idx)[0], torch.where(query == pad_idx)[1], :] = self.negative_inf
        pad_mask[torch.where(key == pad_idx)[0], :, torch.where(key == pad_idx)[1]] = self.negative_inf
        return pad_mask


class Encoder(nn.Module):
    def __init__(self, embedding_dim, n_head, head_dim, feed_forward_dim, n_encoder):
        super().__init__()

        # encoder 쌓기.
        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(embedding_dim, n_head, head_dim, feed_forward_dim) for i in range(n_encoder)]
        )

    def forward(self, x, self_attention_mask):
        # n_encoder 만큼 반복.
        for i in range(len(self.encoder_layers)):
            x = self.encoder_layers[i](x, self_attention_mask)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, embedding_dim, n_head, head_dim, feed_forward_dim):
        """
        Encoder Layer 한층에 대한 구현.
        """
        super().__init__()

        self.multi_head_attention = MultiHeadAttention(embedding_dim, n_head, head_dim)
        self.layer_norm_after_attention = nn.LayerNorm(embedding_dim)
        self.feed_forward = FeedForward(embedding_dim, feed_forward_dim)
        self.layer_norm_after_feedforward = nn.LayerNorm(embedding_dim)

    def forward(self, x, self_attention_mask):
        # multi head attention.
        af_att = self.multi_head_attention(query=x, key=x, value=x, mask=self_attention_mask)

        # residual connection + add & norm.
        sub_output = self.layer_norm_after_attention(x + af_att)  # (batch, seq_len, embed_dim)

        # position-wise feed forward network.
        af_feed = self.feed_forward(sub_output)  # (batch, seq_len, embed_dim)

        # residual connection + add & norm.
        encoder_output = self.layer_norm_after_feedforward(sub_output + af_feed)  # (batch, seq_len, embed_dim)

        return encoder_output


class Decoder(nn.Module):
    def __init__(self, embedding_dim, n_head, head_dim, feed_forward_dim, n_decoder):
        super().__init__()

        # decoder 쌓기.
        self.decoder_layers = nn.ModuleList(
            [DecoderLayer(embedding_dim, n_head, head_dim, feed_forward_dim) for i in range(n_decoder)]
        )

    def forward(self, x, encoder_output, self_attention_mask, encoder_decoder_mask):
        # n_encoder 만큼 반복.
        for i in range(len(self.decoder_layers)):
            x = self.decoder_layers[i](x, encoder_output, self_attention_mask, encoder_decoder_mask)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, embedding_dim, n_head, head_dim, feed_forward_dim):
        super().__init__()

        self.masked_multi_head_attention = MultiHeadAttention(embedding_dim, n_head, head_dim)
        self.layer_norm_after_attention_1 = nn.LayerNorm(embedding_dim)
        self.multi_head_attention = MultiHeadAttention(embedding_dim, n_head, head_dim)
        self.layer_norm_after_attention_2 = nn.LayerNorm(embedding_dim)
        self.feed_forward = FeedForward(embedding_dim, feed_forward_dim)
        self.layer_norm_after_feedforward = nn.LayerNorm(embedding_dim)

    def forward(self, x, encoder_output, self_attention_mask, encoder_decoder_mask):
        """_summary_

        Args:
            x (_type_): Teacher forcing 및 다음 단어 추론을 위한 입력.
            encoder_output (_type_): encoder 모듈의 출력.
            self_attention_mask (_type_): attention mask 적용시 뒤에 단어는 못보도록 처리가 필요함.
            decoder의 중간 feature를 query로 encoder의 key, value 에 대한 attention을 계산.

        Returns:
            _type_: _description_
        """

        # masked multi head attention.
        af_att1 = self.masked_multi_head_attention(query=x, key=x, value=x, mask=self_attention_mask)

        # residual connection + add & norm.
        sub_output = self.layer_norm_after_attention_1(x + af_att1)  # (batch, seq_len, embed_dim)

        # Encoder-decoder multi head attention.
        af_att2 = self.multi_head_attention(
            query=sub_output, key=encoder_output, value=encoder_output, mask=encoder_decoder_mask
        )

        # residual connection + add & norm.
        sub_output = self.layer_norm_after_attention_2(sub_output + af_att2)  # (batch, seq_len, embed_dim)

        # position-wise feed forward network.
        af_feed = self.feed_forward(sub_output)  # (batch, seq_len, embed_dim)

        # residual connection + add & norm.
        decoder_output = self.layer_norm_after_feedforward(sub_output + af_feed)  # (batch, seq_len, embed_dim)

        return decoder_output


class ScaledDotProductAttention(nn.Module):
    def __init__(self, head_dim):
        super().__init__()

        self.head_dim = head_dim

    def forward(self, query_tensor, key_tensor, value_tensor, mask):
        """
        여기서 matmul((seq_len, head_dim), (head_dim, seq_len))이 필요함.
        -> (seq_len, seq_len)으로 각 query 토큰이 다른 key 토큰에 미치는 영향을 볼 수 있음.

        mask: (batch, seq_len, seq_len)
        """
        # (batch, seq_len, n_head, head_dim)

        # (batch, seq_len, n_head, head_dim) -> (batch, n_head, seq_len, head_dim)
        query = query_tensor.permute(0, 2, 1, 3)

        # (batch, seq_len, n_head, head_dim) -> (batch, n_head, head_dim, seq_len)
        key_T = key_tensor.permute(0, 2, 3, 1)

        # (batch, seq_len, n_head, head_dim) -> (batch, n_head, seq_len, head_dim)
        value = value_tensor.permute(0, 2, 1, 3)

        # https://pytorch.org/docs/stable/generated/torch.matmul.html#torch-matmul
        # QK^T / √(d_k)
        # (batch, n_head, seq_len, seq_len)
        correlation = torch.matmul(query, key_T) / self.head_dim ** (1 / 2)

        ## mask 부분을 -inf으로 하는 mask로 변경.
        # (batch, n_head, seq_len, seq_len)
        mask = mask.view(mask.shape[0], 1, mask.shape[1], mask.shape[2]).repeat(1, 6, 1, 1)

        ## attention mask 적용, 0으로 하면 gradient가 사라지므로 작은 값으로 표현.
        # (batch, n_head, seq_len, seq_len)
        correlation_masked = mask + correlation

        ## softmax
        # (batch, n_head, seq_len, seq_len)
        """
        마지막 (seq_len, seq_len) 에서 query(row index)에 대한 key(column index)의 prob으로 표현
        https://www.tablesgenerator.com/text_tables#
        ex)
        +------+-----+------+-----+
        |      | i   | like | you |
        +------+-----+------+-----+
        | i    | 0.2 | 0.5  | 0.3 |
        +------+-----+------+-----+
        | like | 0.1 | 0.2  | 0.7 |
        +------+-----+------+-----+
        | you  | 0.3 | 0.6  | 0.1 |
        +------+-----+------+-----+

        각 행의 합은 1.
        """
        attention = torch.softmax(correlation_masked, dim=-1)

        # matmul and restore shape. (batch, n_head, seq_len, head_dim) -> (batch, seq_len, n_head, head_dim)
        output = torch.matmul(attention, value).permute(0, 2, 1, 3)

        return output


class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim, n_head, head_dim):
        super().__init__()
        """
        Multi head attention에 대한 구현.

        query가 각 key에 대해 미치는 영향(attention)을 학습하고,
        실제 이 값을 value에 곱해서 그 값을 출력한다.
        """

        self.n_head = n_head
        self.head_dim = head_dim
        self.w_query = nn.Linear(embedding_dim, n_head * head_dim)
        self.w_key = nn.Linear(embedding_dim, n_head * head_dim)
        self.w_value = nn.Linear(embedding_dim, n_head * head_dim)
        self.scaled_dot_product_attention = ScaledDotProductAttention(head_dim=head_dim)
        self.scaled_dot_linear = nn.Linear(n_head * head_dim, embedding_dim)

    def forward(self, query, key, value, mask):
        batch_size = query.shape[0]

        # 각 shape은 (batch, seq_len, n_head, head_dim).
        query_tensor = self.w_query(query).view(batch_size, -1, self.n_head, self.head_dim)
        key_tensor = self.w_key(key).view(batch_size, -1, self.n_head, self.head_dim)
        value_tensor = self.w_value(value).view(batch_size, -1, self.n_head, self.head_dim)

        # (batch, seq_len, n_head, head_dim)
        scaled_dot_product_output = self.scaled_dot_product_attention(query_tensor, key_tensor, value_tensor, mask)

        # head들을 concatenate 한다. (batch, seq_len, n_head, head_dim) -> (batch, seq_len, n_head * head_dim)
        scaled_dot_product_output = scaled_dot_product_output.reshape(
            scaled_dot_product_output.shape[0], -1, self.n_head * self.head_dim
        )

        # Dense Layer를 통과시켜 입력 shape과 동일하게 변경.
        output = self.scaled_dot_linear(scaled_dot_product_output)

        return output


class FeedForward(nn.Module):
    def __init__(self, embedding_dim, feed_forward_dim):
        super().__init__()

        # 논문에서는 kernel size가 1인 conv를 2번 적용해도 좋다고 함.
        self.layer1 = nn.Linear(embedding_dim, feed_forward_dim)
        self.layer2 = nn.Linear(feed_forward_dim, embedding_dim)

    def forward(self, x):
        output1 = self.layer1(x)  # (batch, seq_len, feed_forward_dim)
        output2 = self.layer2(output1)  # (batch, seq_len, embed_dim)

        return output2
