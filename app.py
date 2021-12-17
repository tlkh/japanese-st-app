from typing import DefaultDict
import streamlit as st
import spacy
import pykakasi
import requests

ENDPOINT = "https://translate-34x4ouuclq-as.a.run.app/predict"


@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def load_jp_nlp():
    #en_nlp = spacy.load("en_core_web_sm")
    jp_nlp = spacy.load("ja_core_news_sm")
    kks = pykakasi.kakasi()
    #jam = Jamdict()
    return jp_nlp, kks


@st.cache
def make_api_query(src, tgt, text, endpoint):
    query_json = {
        "src": str(src).strip(),
        "tgt": str(tgt).strip(),
        "text": str(text).strip(),
    }
    response = requests.post(endpoint, json=query_json).json()
    return response


def translate_en_to_jp(in_text):
    if len(in_text) > 0:
        src = "en_XX"
        tgt = "ja_XX"
        translation = make_api_query(src, tgt, in_text, ENDPOINT)[
            "translation"]
        translation = translation.strip()
    else:
        translation = ""
    return translation


def translate_jp_to_en(in_text):
    if len(in_text) > 0:
        src = "ja_XX"
        tgt = "en_XX"
        translation = make_api_query(src, tgt, in_text, ENDPOINT)[
            "translation"]
        translation = translation.strip()
    else:
        translation = ""
    return translation


def main():
    jp_nlp, kks = load_jp_nlp()
    direction = st.selectbox(label="Direction", options=[
                             "EN to JP", "JP to EN"])
    in_text = st.text_input(label="Input", value="").strip()
    if direction == "EN to JP":
        translation = translate_en_to_jp(in_text)
        result = kks.convert(translation)
        text_furigana = []
        for item in result:
            text_furigana.append(
                "<ruby>{}<rt>{}</rt></ruby>".format(item['orig'], item['hepburn']))
        text_furigana = "<big>"+"&nbsp;&nbsp;".join(text_furigana)+"</big>"
        st.markdown(text_furigana, unsafe_allow_html=True)

        doc = jp_nlp(translation)
        if len(doc) > 0:
            entity_breakdown = ["|Word|POS|Hiragana|Romaji|", "|-|-|-|-|"]
            for token in doc:
                text = token.text
                meaning = ""
                item = kks.convert(text)[0]
                spam = "|"+item['orig']+"|"+token.pos_ + \
                    "|"+item['hira']+"|"+item['hepburn']+"|"
                entity_breakdown.append(spam)
            entity_breakdown = "\n".join(entity_breakdown)
            st.markdown(entity_breakdown)
    else:
        # JP to EN
        translation = translate_jp_to_en(in_text)
        st.markdown(translation)
        result = kks.convert(in_text)
        text_furigana = []
        for item in result:
            text_furigana.append(
                "<ruby>{}<rt>{}</rt></ruby>".format(item['orig'], item['hepburn']))
        text_furigana = "<big>"+"&nbsp;&nbsp;".join(text_furigana)+"</big>"
        st.markdown(text_furigana, unsafe_allow_html=True)

        doc = jp_nlp(in_text)
        if len(doc) > 0:
            entity_breakdown = ["|Word|POS|Hiragana|Romaji|", "|-|-|-|-|"]
            for token in doc:
                text = token.text
                item = kks.convert(text)[0]
                spam = "|"+item['orig']+"|"+token.pos_ + \
                    "|"+item['hira']+"|"+item['hepburn']+"|"
                entity_breakdown.append(spam)
            entity_breakdown = "\n".join(entity_breakdown)
            st.markdown(entity_breakdown)


if __name__ == "__main__":
    main()
