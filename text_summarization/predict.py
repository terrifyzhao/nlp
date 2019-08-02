from textrank4zh import TextRank4Sentence
from seq2seq import model, gen_sent


def predict_tr(text):
    tr4s = TextRank4Sentence()
    tr4s.analyze(text=text, lower=True, source='all_filters')

    result = tr4s.get_key_sentences(num=3)[0].sentence

    return result


def predict(text):
    result = gen_sent(text, model)

    return result


if __name__ == '__main__':
    text = '海外网7月26日电 香港警方25日在湾仔区拘捕一名23岁梁姓本地男子，被捕男子涉嫌于7月14日在沙田一商场内袭击两名' \
           '警务人员。港媒称，该男子被指有黑帮及贩毒背景，还疑涉及于本月21日冲击中联办并与警方对抗。综合“橙新闻”等港媒报道，' \
           '警方商业罪案调查科总督察陈国伟表示，警方当晚驱散部分非法集结人士时，在商场内有人多次袭警，有警员被包围及袭击，' \
           '包括拳打脚踢、雨伞刺头，导致警员身体严重受伤。陈国伟称，两名警员住院5天后出院，一人面、鼻部骨折，左脸及鼻瘀伤，' \
           '左眼血肿；另一人右后脑有裂伤，需要缝3针，同时全身多处不同程度瘀伤。警方根据现场调查，以及相关人证物证，锁定疑犯' \
           '并作出拘捕，被捕男子则无资料显示受伤。'
    print(predict(text))
