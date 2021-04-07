import numpy as np

from decoders.ctcdecoders.swig_wrapper import (Scorer,
                                               ctc_beam_search_decoder_batch,
                                               ctc_greedy_decoder)


def decode_batch_greedy(probs_split, vocab_list):
    # 批量贪婪解码，参数含义参见./decoders/ctcdecoders/swig_wrapper.py
    results = []
    for i, probs in enumerate(probs_split):
        output_transcription = ctc_greedy_decoder(probs_seq=probs,
                                                  vocabulary=vocab_list)
        results.append(output_transcription)
    return results


def decode_batch_beam_search(probs_split,
                             vocab_list,
                             scorer,
                             beam_size=5,
                             num_processes=16,
                             cutoff_prob=1.0,
                             cutoff_top_n=40):
    # 批量束搜索解码，参数含义参见./decoders/ctcdecoders/swig_wrapper.py
    num_processes = min(num_processes, len(probs_split))
    beam_search_results = ctc_beam_search_decoder_batch(
        probs_split=probs_split,
        vocabulary=vocab_list,
        ext_scoring_func=scorer,
        beam_size=beam_size,
        num_processes=num_processes,
        cutoff_prob=cutoff_prob,
        cutoff_top_n=cutoff_top_n)
    results = [result[0][1] for result in beam_search_results]
    return results


if __name__ == '__main__':
    # probs_split为一个list，每个元素shape为(序列长度，单词个数+1)，且每个时间步经过归一化
    # 注意：blank的索引号等于单词个数
    probs_split = [np.array([[0.6, 0.3, 0.1], [0.5, 0.4, 0.1], [0.1, 0.1, 0.8], [0.3, 0.4, 0.3]])]
    # 每个元素的索引号要与训练时标签对应的索引号一致，注意vocab_list不包含blank
    vocab_list = ['a', 'b']

    # 批量贪婪解码
    greedy_result = decode_batch_greedy(probs_split=probs_split,
                                        vocab_list=vocab_list)
    print(f'贪婪解码：{greedy_result}')

    # 批量束搜索解码
    scorer = Scorer(alpha=1.0,
                    beta=0.3,
                    model_path='./zh_giga.no_cna_cmn.prune01244.klm',
                    vocabulary=vocab_list)
    beamsearch_result = decode_batch_beam_search(probs_split=probs_split,
                                                 vocab_list=vocab_list,
                                                 scorer=scorer)
    print(f'束搜索解码：{beamsearch_result}')
