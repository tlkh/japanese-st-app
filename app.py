from typing import DefaultDict
import streamlit as st
import torch
from transformers import MBartForConditionalGeneration, MBart50Tokenizer
import spacy
import pykakasi


@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def load_translation_model():
    with st.spinner("Loading translation model..."):
        model_name = "facebook/mbart-large-50-many-to-many-mmt"
        model = MBartForConditionalGeneration.from_pretrained(model_name)
        model = model.eval()
        tokenizer = MBart50Tokenizer.from_pretrained(model_name)
    st.success("Done!")
    return tokenizer, model


@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def load_jp_nlp():
    #en_nlp = spacy.load("en_core_web_sm")
    jp_nlp = spacy.load("ja_core_news_sm")
    kks = pykakasi.kakasi()
    #jam = Jamdict()
    return jp_nlp, kks


def translate_en_to_jp(in_text, tokenizer, model):
    if len(in_text) > 0:
        tokenizer.src_lang = "en_XX"
        encoded = tokenizer(in_text, return_tensors="pt")
        with torch.no_grad():
            generated = model.generate(**encoded,
                                       forced_bos_token_id=tokenizer.lang_code_to_id["ja_XX"])
        translation = tokenizer.batch_decode(
            generated, skip_special_tokens=True)[0]
        translation = translation.strip()
    else:
        translation = ""
    return translation

def translate_jp_to_en(in_text, tokenizer, model):
    if len(in_text) > 0:
        tokenizer.src_lang = "ja_XX"
        encoded = tokenizer(in_text, return_tensors="pt")
        with torch.no_grad():
            generated = model.generate(**encoded,
                                       forced_bos_token_id=tokenizer.lang_code_to_id["en_XX"])
        translation = tokenizer.batch_decode(
            generated, skip_special_tokens=True)[0]
        translation = translation.strip()
    else:
        translation = ""
    return translation


def main():
    tokenizer, model = load_translation_model()
    jp_nlp, kks = load_jp_nlp()
    direction = st.selectbox(label="Direction", options=[
                             "EN to JP", "JP to EN"])
    in_text = st.text_input(label="Input", value="").strip()
    if direction == "EN to JP":
        translation = translate_en_to_jp(in_text, tokenizer, model)
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
        translation = translate_jp_to_en(in_text, tokenizer, model)
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
                meaning = ""
                item = kks.convert(text)[0]
                spam = "|"+item['orig']+"|"+token.pos_ + \
                    "|"+item['hira']+"|"+item['hepburn']+"|"
                entity_breakdown.append(spam)
            entity_breakdown = "\n".join(entity_breakdown)
            st.markdown(entity_breakdown)


if __name__ == "__main__":
    main()
