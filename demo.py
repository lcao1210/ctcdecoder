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


def decode_batch_beam_search(probs_split, vocab_list, beam_size, num_processes,
                             scorer, cutoff_prob, cutoff_top_n):
    # 批量束搜索解码，参数含义参见./decoders/ctcdecoders/swig_wrapper.py
    num_processes = min(num_processes, len(probs_split))
    beam_search_results = ctc_beam_search_decoder_batch(
        probs_split=probs_split,
        vocabulary=vocab_list,
        beam_size=beam_size,
        num_processes=num_processes,
        ext_scoring_func=scorer,
        cutoff_prob=cutoff_prob,
        cutoff_top_n=cutoff_top_n)
    results = [result[0][1] for result in beam_search_results]
    return results


if __name__ == '__main__':
    # probs_split为一个list，每个元素shape为(seq_len，label_num)，且每个时间步经过归一化
    probs_split = [np.array([[0.6, 0.4], [0.2, 0.8], [0.3, 0.7]])]
    # 每个元素的索引号要与训练时标签对应的索引号一致
    vocab_list = ['_', 'a', 'b', 'c']

    # 批量贪婪解码
    result_transcript = decode_batch_greedy(probs_split=probs_split,
                                            vocab_list=vocab_list)
    # 批量束搜索解码
    scorer = Scorer(
        alpha=1.0,
        beta=0.3,
        language_model_path='./lm/zh_giga.no_cna_cmn.prune01244.klm',
        vocab_list=vocab_list)
    result_transcript = decode_batch_beam_search(probs_split=probs_split,
                                                 vocab_list=vocab_list,
                                                 beam_size=5,
                                                 num_processes=16,
                                                 scorer=scorer,
                                                 cutoff_prob=1.0,
                                                 cutoff_top_n=40)
