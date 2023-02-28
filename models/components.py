import torch
from torch import nn
import numpy as np


class Transformer(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_head, head_dim, feed_forward_dim, n_encoder):
        super().__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.input_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.encoder = Encoder(embedding_dim, n_head, head_dim, feed_forward_dim, n_encoder)

    def forward(self, x, pad_mask):
        # embedding.
        x_embeded = self.input_embedding(x)  # (batch, seq_len, embed_dim)

        # positional encoding.
        pos_encoding = self.postional_encoding(x, self.embedding_dim, pad_mask)
        inputs = x_embeded + torch.FloatTensor(pos_encoding)  # (batch, seq_len, embed_dim)

        # encoder module.
        encoder_output = self.encoder(inputs, pad_mask)  # (batch, seq_len, embed_dim)

        return 0

    def postional_encoding(self, input, embedding_dim, pad_mask):
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

        # padding 부분은 0을 적용.
        encoding_matrix[pad_mask[0], pad_mask[1], :] = 0

        return encoding_matrix


class Encoder(nn.Module):
    def __init__(self, embedding_dim, n_head, head_dim, feed_forward_dim, n_encoder):
        super().__init__()

        # encoder 쌓기.
        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(embedding_dim, n_head, head_dim, feed_forward_dim) for i in range(n_encoder)]
        )

    def forward(self, x, pad_mask):
        # n_encoder 만큼 반복.
        for i in range(len(self.encoder_layers)):
            x = self.encoder_layers[i](x, pad_mask)
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

    def forward(self, x, pad_mask):
        # multi head attention.
        af_att = self.multi_head_attention(query=x, key=x, value=x, query_mask=pad_mask, key_mask=pad_mask)

        # residual connection + add & norm.
        sub_output = self.layer_norm_after_attention(x + af_att)  # (batch, seq_len, embed_dim)

        # position-wise feed forward network.
        af_feed = self.feed_forward(sub_output)  # (batch, seq_len, embed_dim)

        # residual connection + add & norm.
        encoder_output = self.layer_norm_after_feedforward(sub_output + af_feed)  # (batch, seq_len, embed_dim)

        return encoder_output


class ScaledDotProductAttention(nn.Module):
    def __init__(self, head_dim):
        super().__init__()

        self.head_dim = head_dim

    def forward(self, query_tensor, key_tensor, value_tensor, query_mask, key_mask):
        """
        여기서 matmul((seq_len, head_dim), (head_dim, seq_len))이 필요함.
        -> (seq_len, seq_len)으로 각 토큰이 다른 토큰에 미치는 영향을 볼 수 있음.

        query_mask: (array(batch_idx), array(seq_idx))
        key_mask: (array(batch_idx), array(seq_idx))
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

        ## mask 부분을 0으로 하는 mask 생성.
        attention_mask = torch.ones(correlation.shape)
        attention_mask[query_mask[0], :, query_mask[1], :] = 0
        attention_mask[key_mask[0], :, :, key_mask[1]] = 0

        ## attention mask, 0으로 하면 gradient가 사라지므로 작은 값으로 표현.
        # (batch, n_head, seq_len, seq_len)
        correlation_masked = correlation.masked_fill(attention_mask == 0, -1e10)

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

    def forward(self, query, key, value, query_mask, key_mask):
        batch_size = query.shape[0]

        # 각 shape은 (batch, seq_len, n_head, head_dim).
        query_tensor = self.w_query(query).view(batch_size, -1, self.n_head, self.head_dim)
        key_tensor = self.w_key(key).view(batch_size, -1, self.n_head, self.head_dim)
        value_tensor = self.w_value(value).view(batch_size, -1, self.n_head, self.head_dim)

        # (batch, seq_len, n_head, head_dim)
        scaled_dot_product_output = self.scaled_dot_product_attention(
            query_tensor, key_tensor, value_tensor, query_mask, key_mask
        )

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
