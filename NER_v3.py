import joblib
import spacy
from spacy.tokens import Doc
import streamlit as st
import matplotlib.pyplot as plt
from collections import Counter
import random
import numpy as np
import pandas as pd
import plotly.express as px


model = joblib.load('model/model.joblib')

stopwords = ["ผู้", "ที่", "ซึ่ง", "อัน"]

N_SHUFFLE = 5
N_SHUFFLE_SUMMARY = 100


def tokens_to_features(tokens, i):
    word = tokens[i]
    features = {
        "bias": 1.0,
        "word.word": word,
        "word[:3]": word[:3],
        "word.isspace()": word.isspace(),
        "word.is_stopword()": word in stopwords,
        "word.isdigit()": word.isdigit(),
        "word.islen5": word.isdigit() and len(word) == 5
    }
    if i > 0:
        prevword = tokens[i - 1]
        features.update({
            "-1.word.prevword": prevword,
            "-1.word.isspace()": prevword.isspace(),
            "-1.word.is_stopword()": prevword in stopwords,
            "-1.word.isdigit()": prevword.isdigit(),
        })
    else:
        features["BOS"] = True
    if i < len(tokens) - 1:
        nextword = tokens[i + 1]
        features.update({
            "+1.word.nextword": nextword,
            "+1.word.isspace()": nextword.isspace(),
            "+1.word.is_stopword()": nextword in stopwords,
            "+1.word.isdigit()": nextword.isdigit(),
        })
    else:
        features["EOS"] = True
    return features

TAG_COLORS = {
    "O": "#99ff99",
    "ADDR": "#ffadad",
    "LOC": "#fdffb6",
    "POST": "#9ce7d5"
}


TAG_COLORS_VERSION_DEAR = {
    "O": "#FFC0CB",
    "ADDR": "#ADD8E6",
    "LOC": "#fdffb6",
    "POST": "#9ce7d5"
}

NO_TAG_COLOR = '#f0f0f0'

def create_token_version_pson(token: str, entity_label: str = None):

    base_html = """
        <style>
        .entity {
            display: inline-block;
            padding: 0.3em 0.6em;
            margin: 0.1em;
            border-radius: 0.5em;
            line-height: 1.2;
            border: 1px solid #ddd;
        }
        .label {
            font-size: 0.8em;
            color: #ffffff;
            padding: 0.2em 0.4em;
            border-radius: 0.3em;
            margin-left: 0.3em;
            vertical-align: middle;
        }
        .empty-border {
            border: 1px solid #ddd;
            background-color: transparent !important;
        }
        .highlighted {
            border: 1px solid transparent;
        }
        </style>

    """
    if entity_label is None:
        return base_html + f"<span class='entity empty-border'>{token}</span> "
    else:
        color = TAG_COLORS.get(entity_label, "#ffffff")
        return base_html + f"<span class='entity highlighted' style='background-color: {color};'>{token}<span class='label' style='background-color: #333;'>{entity_label}</span></span> "
    
def create_token_tag_version_dear(token: str, entity_label: str = None) -> str:
    # Define colors for each label and highlighted words
    highlight_color = "#FFD700"  # Gold for highlighted words

    color = TAG_COLORS_VERSION_DEAR.get(entity_label, "#ffffff")

    # Initialize HTML output
    html_output = '<div style="font-family: sans-serif; text-align: left; line-height: 1.5;">'

    if entity_label is None:
        html_output += f'<div style="width: 100%; display: inline-block; margin: 0 5px; text-align: center; border: 1px solid {color}; background-color: {NO_TAG_COLOR}; padding: 5px; border-radius: 5px;">'
        html_output += f'<div style="color: black; padding: 2px 5px; margin-top: 2px; border-radius: 3px; text-overflow: ellipsis;">{token}</div>'
        # html_output += f'<div style="display: inline-block; margin: 0 5px; text-align: center; padding: 5px;">'
        # html_output += f'<div>{token}</div>'
        html_output += '</div>'
    else:
        html_output += f'<div style="width: 100%; display: inline-block; margin: 0 5px; text-align: center; border: 1px solid {color}; background-color: {color}; padding: 5px; border-radius: 5px;">'
        html_output += f'<div style="color: black; padding: 2px 5px; margin-top: 2px; border-radius: 3px; text-overflow: ellipsis;">{token}</div>'
        html_output += f'<div style="background-color: white; color: #6D6875; padding: 2px 5px; margin-top: 2px; border-radius: 3px; font-weight: bold; text-overflow: ellipsis;">{entity_label}</div>'
        html_output += '</div>'
    
    html_output += '</div>'
    return html_output

create_token_tag = create_token_tag_version_dear
    
def parse(text: str):
    tokens = text.split()
    features = [tokens_to_features(tokens, i) for i in range(len(tokens))]
    predictions = model.predict([features])[0]
    return tokens, predictions

def make_result_df(tokens, predictions):
    return pd.DataFrame({
        'index': np.arange(len(tokens)),
        'token': tokens,
        'tag': predictions,
    })

def parse_and_visualize(text, selected_entities, highlighted_words, is_initial=False):
    # tokens = text.split()
    # features = [tokens_to_features(tokens, i) for i in range(len(tokens))]
    # predictions = model.predict([features])[0]
    tokens, predictions = parse(text)

    nlp = spacy.blank("th")
    doc = Doc(nlp.vocab, words=tokens)


    entity_counts = Counter(predictions)
    entity_types = ["ADDR", "LOC", "POST", "O"]
    # counts = {entity: entity_counts.get(entity, 0) for entity in entity_types}

    result_df = make_result_df(tokens, predictions)

    n_word = len(doc)
    column_spec = np.full(n_word, 1 / n_word)

    columns = st.columns(column_spec)

    for i, token in enumerate(doc):
        entity_label = predictions[i]
        if is_initial:  # ถ้าเป็นการ analyze ครั้งแรก
            if entity_label in selected_entities:
                token_tag = create_token_tag(token.text, entity_label)
            else:
                token_tag = create_token_tag(token.text)
        else:  # ถ้าเป็นการ shuffle
            if token.text in highlighted_words and entity_label in selected_entities:
                token_tag = create_token_tag(token.text, entity_label)
            else:
                token_tag = create_token_tag(token.text)

        columns[i].markdown(token_tag, unsafe_allow_html=True)
        
    return result_df


def shuffle_text(text, seed: int = 7):
    words = text.split()
    # print('words', words)
    # random.shuffle(words)
    np.random.RandomState(seed).shuffle(words)
    # print('shuffled words', words)
    return ' '.join(words)

def create_dataframe_result(data):
    df_result_counter = pd.DataFrame.from_dict(data, orient='index').reset_index()
    df_result_counter.columns = ['Class', 'Count']
    return df_result_counter

st.set_page_config(layout="wide")
# สร้าง UI
st.title("[What if analysis] - If the address is SHUFFLED !")

text_input_section, submit_button = st.columns(spec=[0.8, 0.2], vertical_alignment='bottom')
with text_input_section:
    # Input text
    text = st.text_input("Text Input:", value=None)
with submit_button:
    # Analyze
    if st.button("Analyze !"):
    # if text:
        st.session_state.is_analyzed = True
        st.session_state.show_word_selection = False   
        st.session_state.shuffled_texts = []

# Session state initialization
if 'initial_result' not in st.session_state:
    st.session_state.initial_result = None
if 'shuffled_texts' not in st.session_state:
    st.session_state.shuffled_texts = []

# Sidebar filters
# selected_entities = st.sidebar.multiselect(
#     "Named Entities",
#     options=["ADDR", "LOC", "POST", "O"],
#     default=["ADDR", "LOC", "POST", "O"]
# )



if 'is_analyzed' in st.session_state and st.session_state.is_analyzed:
    what_if_tab, summary_tab = st.tabs(['What If Analysis - Shuffle', 'Summary'])

    with what_if_tab:
        tag_selector_section, token_selector_section = st.columns(
            spec=[0.2, 0.8]
        )

        with tag_selector_section:
            selected_entities = st.pills(
                'Named Entities',
                options=["ADDR", "LOC", "POST", "O"],
                default=["ADDR", "LOC", "POST", "O"],
    #             help='''
    # ADDR : Address

    # LOC : Location

    # POST : Postal Code

    # O : Others
    #             ''',
                help='''
LOC : Sub-district, District, or Provinct

POST : Postal Code

ADDR : Other address element

O : Other
                ''',
                selection_mode='multi'

            )
        if text:
            st.markdown("## Original Prediction:")
            st.session_state.initial_result = text
            result_df = parse_and_visualize(text, selected_entities, [],is_initial=True)
            st.session_state.ner_done = True
        else:
            st.warning("Please enter text for analysis.")

        # Shuffle button and functionality
        if 'ner_done' in st.session_state and st.session_state.ner_done:
            st.markdown('')
            shuffled = st.button("Shuffle Text")

            if shuffled:
                #st.write("Shuffled Texts:")
                st.session_state.shuffled_texts = [shuffle_text(text, seed=None) for i in range(N_SHUFFLE)]
                #for shuffled_text in st.session_state.shuffled_texts:
                    #st.text(shuffled_text)
                    #st.write('----------------------------------------')
                
                # Enable word selection
                st.session_state.show_word_selection = True


        
            if hasattr(st.session_state, 'show_word_selection') and st.session_state.show_word_selection:
            # if hasattr(st.session_state, 'is_shuffled') and st.session_state.is_shuffled:


                with what_if_tab:
                    all_results = []
                    original_words = text.split()

                    # st.sidebar.multiselect(
                    #     "Choose Words to Highlight",
                    #     options=original_words,
                    #     default=[]
                    # )

                    with token_selector_section:
                        # highlighted_words = st.pills(
                        #     'Choose Words to Highlight',
                        #     options=original_words,
                        #     default=[],
                        #     selection_mode='multi'
                        # )
                        pass
                    # highlighted_words = highlighted_words or list()
                
                    # if highlighted_words or True:  # แสดงทุกครั้งแม้ยังไม่มีการเลือกคำ
                    if True:
                        
                        st.markdown("## Shuffle !")
                        highlighted_words = st.pills(
                            'Choose Words to Highlight',
                            options=original_words,
                            default=[],
                            selection_mode='multi'
                        )

                        for shuffle_id, shuffled_text in enumerate(st.session_state.shuffled_texts):
                            result_df = parse_and_visualize(shuffled_text, selected_entities, highlighted_words,is_initial=False)
                            
                            all_results.append(result_df.assign(shuffle_id=shuffle_id))
                            st.write('----------------------------------------')



        else:
            st.write(" ")

    with summary_tab:
        original_tokens, original_predictions = parse(text)
        original_result_df = make_result_df(original_tokens, original_predictions)
        original_result_df['order'] = 'Index: ' + (original_result_df['index'] + 1).astype(str)
        original_result_df.sort_values('index', inplace=True)

        all_results = []
        for shuffle_id in range(N_SHUFFLE_SUMMARY):
            shuffled_text = shuffle_text(text, seed=shuffle_id)
            tokens, predictions = parse(shuffled_text)
            result_df = make_result_df(tokens, predictions)
            all_results.append(result_df.assign(shuffle_id=shuffle_id))

        selected_word = st.pills(
            'Word',
            options=original_tokens,
            default=original_tokens[0],
            selection_mode='single'
        )

        # st.write("Original Text:")
        # st.write(selected_word)

        all_result_df = pd.concat(all_results)
        all_result_df['order'] = 'Index: ' + (all_result_df['index'] + 1).astype(str)

        word_result_df = all_result_df[all_result_df['token'] == selected_word]

        df = word_result_df.groupby(['index', 'order', 'tag']).agg(count=('token', 'count')).reset_index()
        df['count_all_word'] = df.groupby('order')['count'].transform('sum')
        df['percentage'] = df['count'] / df['count_all_word'] * 100
        df.sort_values(['index', 'tag'])

        # st.write(all_result_df)
        # st.write(df)

        fig = px.bar(
            df,
            x='order',
            y='percentage',
            color='tag',
            barmode='stack',
            text='tag',
            color_discrete_map=TAG_COLORS_VERSION_DEAR,
            category_orders={
                'tag': ['ADDR', 'LOC', 'POST', 'O'],
                'order': original_result_df['order']
            },
            
        )
        fig.update_layout(
            font=dict(size=20),
            yaxis=dict(title='Probability (%)'),
            xaxis=dict(title='Shuffled Order'),
        )

        st.markdown(f'# What if "{selected_word}" is shuffled ?')
        st.markdown(f'###### What "{selected_word}" gonna be ?')
        st.plotly_chart(fig, theme=None)