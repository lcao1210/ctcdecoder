from decoders.ctcdecoders.swig_wrapper import Scorer
from decoders.ctcdecoders.swig_wrapper import ctc_beam_search_decoder_batch
from decoders.ctcdecoders.swig_wrapper import ctc_greedy_decoder


class Model():
    def __init__(self, vocab_list):
        self.vocab_list = vocab_list  # 字符集

    def decode_batch_greedy(self, probs_split):
        # 参数含义参见./decoders/ctcdecoders/swig_wrapper.py
        results = []
        for i, probs in enumerate(probs_split):
            output_transcription = ctc_greedy_decoder(
                probs_seq=probs, vocabulary=self.vocab_list)
            results.append(output_transcription)
        return results

    def decode_batch_beam_search(self, probs_split, beam_size, num_processes,
                                 scorer, cutoff_prob, cutoff_top_n):
        # 参数含义参见./decoders/ctcdecoders/swig_wrapper.py
        num_processes = min(num_processes, len(probs_split))
        beam_search_results = ctc_beam_search_decoder_batch(
            probs_split=probs_split,
            vocabulary=self.vocab_list,
            beam_size=beam_size,
            num_processes=num_processes,
            ext_scoring_func=scorer,
            cutoff_prob=cutoff_prob,
            cutoff_top_n=cutoff_top_n)
        results = [result[0][1] for result in beam_search_results]
        return results


if __name__ == '__main__':
    probs_split = [[0.6, 0.4], [0.2, 0.8], [0.3, 0.7]]
    vocab_list = ['_', 'a', 'b', 'c']

    model = Model(vocab_list=vocab_list)
    # 贪婪解码
    result_transcript = model.decode_batch_greedy(probs_split=probs_split)
    # 束搜索解码
    scorer = Scorer(
        alpha=1.0,
        beta=0.3,
        language_model_path='./lm/zh_giga.no_cna_cmn.prune01244.klm',
        vocab_list=model.vocab_list)
    result_transcript = model.decode_batch_beam_search(probs_split=probs_split,
                                                       beam_size=5,
                                                       num_processes=16,
                                                       scorer=scorer,
                                                       cutoff_prob=1.0,
                                                       cutoff_top_n=40)
