from jieba import analyse
from wordcloud import WordCloud

font_path = '/System/Library/Fonts/Hiragino Sans GB.ttc'


def get_word(text, mode='text_rank', topK=10, image_path='word.jpg'):
    method = None
    if mode == 'text_rank':
        method = analyse.textrank
    elif mode == 'tf_idf':
        method = analyse.extract_tags
    word = method(text, topK=topK)
    wordcloud = WordCloud(font_path=font_path, background_color='white').generate(' '.join(word))
    wordcloud.to_file(image_path)
    return word


if __name__ == '__main__':
    text = ''
    with open('../data/text_summarization/train.csv', encoding='utf-8')as file:
        for line in file.readlines()[0:100]:
            text = text + line.strip()
    word = get_word(text)
    print(word)
