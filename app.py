import streamlit as st
import pinecone
import openai
from openai.embeddings_utils import get_embedding
import json
#import pandas as pd

OPENAI_KEY = st.secrets["OPENAI_KEY"]
PINECONE_KEY = st.secrets["PINECONE_KEY"]
INDEX = st.secrets["INDEX"]

instructions = {
    "conservative q&a": "Answer the question based on the context below, and if the question can't be answered based on the context, say \"I don't know\"\n\nContext:\n{0}\n\n---\n\nQuestion: {1}\n\nAnswer:",
    "paragraph about a question": "Write a paragraph based on the context below, and if the question can't be answered based on the context, say \"I don't know\"\n\nContext:\n{0}\n\n---\n\nQuestion: {1}\n\nAnswer:",
    "bullet points": "Write a bullet point list based on the context below, and if the question can't be answered based on the context, say \"I don't know\"\n\nContext:\n{0}\n\n---\n\nQuestion: {1}\n\nAnswer:"
}


@st.cache_resource(show_spinner=False)
def init_openai():
    # initialize connection to OpenAI
    openai.api_key = OPENAI_KEY


@st.cache_resource(show_spinner=False)
def init_key_value():
    with open('./embeddings/mapping.json', 'r') as fp:
        mappings = json.load(fp)
    return mappings


@st.cache_resource(show_spinner=False)
def init_pinecone(index_name):
    # initialize connection to Pinecone vector DB (app.pinecone.io for API key)
    pinecone.init(
        api_key=PINECONE_KEY,
        environment="us-east1-gcp"  # find next to API key in console
    )
    index = pinecone.Index(index_name)
    stats = index.describe_index_stats()
    dims = stats['dimension']
    count = stats['namespaces']['']['vector_count']
    return index, dims, count


def create_context(question, index, mappings, lib_meta, max_len=3750, top_k=5):
    """
    Find most relevant context for a question via Pinecone search
    """
    q_embed = get_embedding(question, engine=f'text-embedding-ada-002')
    #df = pd.DataFrame(lib_meta)
    #print(df)
    res = index.query(
        q_embed, top_k=top_k,
        include_metadata=True, filter={
            'source': {'$in': lib_meta}
            # 'source': {'$in': ["Contract Management", "Contracting-Drilling and Well Services", "Due Diligence - New Market Entry", "Proposal Management Process"]}
        })
    cur_len = 0
    contexts = []
    sources = []

    for row in res['matches']:
        text = mappings[row['id']]
        cur_len += row['metadata']['n_tokens'] + 4
        if cur_len < max_len:
            contexts.append(text)
            sources.append(row['metadata'])
        else:
            cur_len -= row['metadata']['n_tokens'] + 4
            if max_len - cur_len < 200:
                break
    # df = pd.DataFrame(sources)
    # print(df)
    return "\n\n###\n\n".join(contexts), sources


def answer_question(
    index,
    mappings,
    fine_tuned_qa_model="text-davinci-003",
    question="What is the goal of the PW approval process?",
    instruction="Answer the question based on the context below, and if the question can't be answered based on the context, say \"I don't know\"\n\nContext:\n{0}\n\n---\n\nQuestion: {1}\nAnswer:",
    max_len=3550,
    size="curie",
    top_k=5,
    debug=False,
    max_tokens=400,
    stop_sequence=None,
    domains=["contract management.txt", "contracting-drilling & well services.txt",
             "due diligence - new market entry.txt", "proposal management process.txt"],
):
    """
    Answer a question based on the most similar context from the dataframe texts
    """
    context, sources = create_context(
        question,
        index,
        mappings,
        lib_meta=domains,
        max_len=max_len,
        top_k=top_k
    )
    # if debug:
    # print("Context:\n" + context)
    # print("\n\n")
    try:
        # fine-tuned models requires model parameter, whereas other models require engine parameter
        model_param = (
            {"model": fine_tuned_qa_model}
            if ":" in fine_tuned_qa_model
            and fine_tuned_qa_model.split(":")[1].startswith("ft")
            else {"engine": fine_tuned_qa_model}
        )
        # print(instruction.format(context, question))
        response = openai.Completion.create(
            prompt=instruction.format(context, question),
            temperature=0,
            max_tokens=max_tokens,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=stop_sequence,
            **model_param
        )
        # usage_dict = json.loads(response)
        # print(response["usage"])
        return response["choices"][0]["text"].strip(), response["usage"],  sources, instruction.format(context, question)
    except Exception as e:
        print(e)
        return ""


def search(index, text_map, query, style, top_k, lib_filters):
    if query != "":
        with st.spinner("Retrieving, please wait..."):
            answer, tokens, sources, context = answer_question(
                index, text_map,
                question=query,
                instruction=instructions[style],
                top_k=top_k
            )
            # lowercase relevant lib filters
            #lib_meta = [lib.lower()
            #            for lib in lib_filters.keys() if lib_filters[lib]]
            #lower_libs = [lib.lower() for lib in libraries]
        # display the answer
        st.write(answer)
        with st.expander("Sources"):
            for source in sources:
                st.write(f"""
                {source['source']})
                """)


st.set_page_config(
    page_title="Parker Wellbore POC",
    page_icon="images/GTL.png",
    layout="wide",
    initial_sidebar_state="expanded"
)


st.markdown("""
<link
  rel="stylesheet"
  href="https://fonts.googleapis.com/css?family=Roboto:300,400,500,700&display=swap"
/>
""", unsafe_allow_html=True)

libraries = [
    "contract management.txt",
    "contracting-drilling & well services.txt",
    "due diligence - new market entry.txt",
    "proposal management process.txt"
]

with st.spinner("Connecting to OpenAI..."):
    retriever = init_openai()

with st.spinner("Connecting to PINECONE..."):
    index, dims, count = init_pinecone(INDEX)
    text_map = init_key_value()


st.write("# Parker Wellbore Document Search ")
search = st.container()
query = search.text_input('Ask a question about a document!', "")

style = st.radio(label='Answer Return Style', options=[
        'Paragraph about a question', 'Conservative Q&A',
        'Bullet points'
    ])

# with search.expander("Search Options"):
#    st.write("""#### Choose Response Style """)
#    style = st.radio(label='Style', options=[
#        'Paragraph about a question', 'Conservative Q&A',
#        'Bullet points'
#    ])

    # add section for filters
#st.write("""
#            #### Documents
#            """)
    # create two cols
#cols = st.columns(2)
    # add filtering based on library
#lib_filters = {}
#lib_filters = libraries
#for lib in libraries:
#        i = len(lib_filters.keys()) % 2
#        with cols[i]:
#            lib_filters[lib] = st.checkbox(lib, value=True)

top_k = st.slider(
        "top_k",
        min_value=1,
        max_value=20,
        value=5
    )

st.write("---")

st.sidebar.write(f"""
    ### Info

    **Pinecone index name**: {INDEX}

    **Pinecone index size**: {count}

    **OpenAI embedding model**: *text-embedding-ada-002*

    **Vector dimensionality**: {dims}

    **OpenAI generation model**: *text-davinci-003*

    ---

    ### How it Works

    This tool has pre-loaded several process documents from Parker Wellbore

    Users can ask questions like **"Explan the contract management process?"** or **"What is the role of the ELT in due diligence?"**, choose a answer style, and return relevant results!
    
    The app is powered using OpenAI's embedding service with Pinecone's vector database. The whole process consists
    of *three* steps:
    
    **1**. Questions are fed into OpenAI's embeddings service to generate a {dims}-dimensional query vector.
    
    **2**. We use Pinecone to identify similar context vectors (previously encoded from Q&A pages).

    **3**. Relevant pages are passed in a new question to OpenAI's generative model, returning our answer.

    ---

    ### Usage

    If you'd like OpenAI to consider more or less pages, try changing the `top_k` slider.  

    The Tabs panel at the bottom displays the **Promt** that was sent to the model, the **Sources** that were useed for the promot and the number of **Tokens** used.
     
    """)

if search.button("Go!") or query != "":
    with st.spinner("Retrieving, please wait..."):
        # lowercase relevant lib filters
        #lib_meta = [lib.lower()
        #            for lib in lib_filters.keys() if lib_filters[lib]]
        lib_meta = libraries
        # ask the question
        #print("style:" + instructions[style.lower()])
        answer, tokens, sources, prompt = answer_question(
            index, text_map,
            question=query,
            instruction=instructions[style.lower()],
            top_k=top_k,
            domains=lib_meta
        )
    # display the answer
    st.write(answer)

    tab_sources, tab_usage, tab_prompt = st.tabs(["Sources", "Tokens", "Prompt"])

    #display the source
    for source in sources:
            tab_sources.write(source['source'])
    
    #display token counts
    tab_usage.write("Completion Tokens:  " + "**" + str(tokens["completion_tokens"]) + "**")
    tab_usage.write("Prompt Tokens:  " + "**" + str(tokens["prompt_tokens"]) + "**")
    tab_usage.write("Total Tokens:  " + "**" + str(tokens["total_tokens"]) + "**")
    total_cost = (tokens["total_tokens"]/1000) *0.0200
    tab_usage.write("Davinci" + "\t\t" + " - v2 - " + "\t$" + '%.8s' %str(total_cost))

    #display the prompt and answer
    tab_prompt.write(prompt)
    tab_prompt.write(answer)
