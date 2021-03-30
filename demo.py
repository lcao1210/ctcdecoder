from decoders.ctcdecoders.swig_wrapper import Scorer
from decoders.ctcdecoders.swig_wrapper import ctc_beam_search_decoder_batch
from decoders.ctcdecoders.swig_wrapper import ctc_greedy_decoder


class Model():
    def __init__(self, alpha, beta, language_model_path, vocab_list,
                 batch_size, cutoff_prob, cutoff_top_n):
        self.alpha = alpha
        self.beta = beta
        self.language_model_path = language_model_path
        self.vocab_list = vocab_list
        self.batch_size = batch_size
        self.cutoff_prob = cutoff_prob
        self.cutoff_top_n = cutoff_top_n
        self.scorer = Scorer(alpha=self.alpha,
                             beta=self.beta,
                             language_model_path=self.language_model_path,
                             vocab_list=self.vocab_list)

    def decode_batch_greedy(self, probs_split, vocab_list):
        results = []
        for i, probs in enumerate(probs_split):
            output_transcription = ctc_greedy_decoder(probs_seq=probs,
                                                      vocabulary=vocab_list)
            results.append(output_transcription)
        return results

    def decode_batch_beam_search(self, probs_split, beam_alpha, beam_beta,
                                 beam_size, cutoff_prob, cutoff_top_n,
                                 vocab_list, num_processes):
        num_processes = min(num_processes, len(probs_split))
        beam_search_results = ctc_beam_search_decoder_batch(
            probs_split=probs_split,
            vocabulary=vocab_list,
            beam_size=beam_size,
            num_processes=num_processes,
            ext_scoring_func=self._ext_scorer,
            cutoff_prob=cutoff_prob,
            cutoff_top_n=cutoff_top_n)

        results = [result[0][1] for result in beam_search_results]
        return results


if __name__ == '__main__':
    model = Model()
    probs_split = [[0.6, 0.4], [0.2, 0.8], [0.3, 0.7]]
    result_transcript = model.decode_batch_greedy(probs_split=probs_split,
                                                  vocab_list=model.vocab_list)
    result_transcript = model.decode_batch_beam_search(
        probs_split=probs_split,
        beam_alpha=model.alpha,
        beam_beta=model.beta,
        beam_size=model.beam_size,
        cutoff_prob=model.cutoff_prob,
        cutoff_top_n=model.cutoff_top_n,
        vocab_list=model.vocab_list,
        num_processes=16)
