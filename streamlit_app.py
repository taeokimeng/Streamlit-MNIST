import streamlit as st
import urllib
import src.mnist_run as mnist_run
from src.resource import OPTIMIZERS, METRICS, SELECTED_OPTIMIZER, SELECTED_METRIC, SELECTED_EPOCH, EPOCH_MIN, EPOCH_MAX

# Streamlit encourages well-structured code, like starting execution in a main() function.
def main():
    st.set_page_config(page_title="Fashion MNIST with Streamlit", layout="wide")

    # Render the readme as markdown using st.markdown.
    readme_text = st.markdown(get_file_content_as_string("instructions.md"))

    # Once we have the dependencies, add a selector for the app mode on the sidebar.
    st.sidebar.title("What to do")
    app_mode = st.sidebar.selectbox("Please choose the app mode",
        ["Show instructions", "Run the app", "Show the source code"])
    if app_mode == "Show instructions":
        st.sidebar.success('To continue select "Run the app".')
    elif app_mode == "Show the source code":
        readme_text.empty()
        st.code(get_file_content_as_string("streamlit_app.py"))
    elif app_mode == "Run the app":
        readme_text.empty()
        SELECTED_OPTIMIZER = optimizer_selector_ui()
        SELECTED_METRIC = metric_selector_ui()
        SELECTED_EPOCH = epoch_selector_ui()
        mnist_run.run_mnist(selected_optimizer=SELECTED_OPTIMIZER, selected_metric=SELECTED_METRIC, selected_epochs=SELECTED_EPOCH)

# Download a single file and make its content available as a string.
@st.cache(show_spinner=False) # Default is True to show a spinner when there is a cache miss.
def get_file_content_as_string(path):
    url = 'https://raw.githubusercontent.com/streamlit/demo-self-driving/master/' + path
    response = urllib.request.urlopen(url)
    return response.read().decode("utf-8")

def optimizer_selector_ui():
    st.sidebar.markdown("# Optimizer")
    optimizer_name = st.sidebar.selectbox("Please choose an optimizer", OPTIMIZERS, 0) # Default selection is Adam
    return optimizer_name

def metric_selector_ui():
    st.sidebar.markdown("# Metric")
    metric_name = st.sidebar.radio("Please choose a metric", list(METRICS.keys()), 0) # Default selection is Accuracy
    return METRICS[metric_name]

def epoch_selector_ui():
    st.sidebar.markdown("# Epoch")
    epochs_selection = st.sidebar.slider("Please choose the epoch", EPOCH_MIN, EPOCH_MAX, EPOCH_MIN)
    return int(epochs_selection)

if __name__ == "__main__":
    main()