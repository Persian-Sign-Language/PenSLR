import Levenshtein as lev

def output_ctc(output):
    ans = ""
    tmp = "" if output[0] == "_" else output[0]
    ans = ans + tmp
    for i in range(len(output)):
        if output[i] != tmp and output[i] != "_":
            tmp = output[i]
            ans = ans + tmp
        if output[i] == "_":
            tmp = output[i]
    return ans

def calculate_wer_similarity(prediction, target):
    prediction = prediction.lower().split()
    target = target.lower().split()

    distance = lev.distance(prediction, target)
    length = len(target)
    wer = float(distance) / float(length)

    return 1 - wer
