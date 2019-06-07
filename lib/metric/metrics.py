import lib


# Calculate f1 score of a corpus.
def f1(inputs, preds, golds, spelling=False):
    assert len(preds) == len(golds) == len(inputs)
    correct_norm, total_norm, total_nsw = 0.0, 0.0, 0.0
    if(spelling): #for spelling we got only one word each time
        for input_token, pred_token, oracle_token in zip(inputs, preds, golds):
            pred_token = ''.join(pred_token)
            if pred_token.lower() != input_token.lower() and oracle_token.lower() == pred_token.lower() and oracle_token.strip():
                correct_norm += 1
            if oracle_token.lower() != input_token.lower() and oracle_token.strip():
                total_nsw += 1
            if pred_token.lower() != input_token.lower() and pred_token.strip():
                total_norm += 1
    else:
        for input_tokens, pred_tokens, oracle_tokens in zip(inputs, preds, golds):
            sent_length = min(len(input_tokens), len(oracle_tokens))
            while len(pred_tokens) < sent_length : pred_tokens.append(lib.constants.PAD_WORD)
            for i in range(sent_length):
                pred_token = pred_tokens[i]
                oracle_token = oracle_tokens[i]
                input_token = input_tokens[i]
                if pred_token.lower() != input_token.lower() and oracle_token.lower() == pred_token.lower() and oracle_token.strip():
                    correct_norm += 1
                if oracle_token.lower() != input_token.lower() and oracle_token.strip():
                    total_nsw += 1
                if pred_token.lower() != input_token.lower() and pred_token.strip():
                    total_norm += 1
    # calc p, r, f
    p = r = f1 = 0.0
    if(total_norm!=0 and correct_norm!= 0): p = correct_norm / total_norm
    if(total_norm!=0 and total_nsw!= 0): r = correct_norm / total_nsw
    if p != 0 and r != 0: f1 =  (2 * p * r) / (p + r)
    results = {}
    results["correct_norm"] = correct_norm
    results["total_norm"] = total_norm
    results["total_nsw"] = total_nsw
    results["precision"] = p
    results["recall"] = r
    results["f1"] = f1
    return results