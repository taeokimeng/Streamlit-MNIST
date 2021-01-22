import streamlit as st
import urllib
from src.mnist_run import *
from src.resource import * # OPTIMIZERS, METRICS, SELECTED_OPTIMIZER, SELECTED_METRIC, SELECTED_EPOCH, EPOCH_MIN, EPOCH_MAX

# Streamlit encourages well-structured code, like starting execution in a main() function.
def main():
    st.set_page_config(page_title="Fashion MNIST with Streamlit", layout="wide")

    # Render the readme as markdown using st.markdown.
    # readme_text = st.markdown(get_file_content_as_string(INTRO_MD))
    sidebar_menu_ui()

def sidebar_menu_ui():
    # Once we have the dependencies, add a selector for the app mode on the sidebar.
    st.sidebar.title(SIDEBAR_TITLE)
    app_mode = st.sidebar.selectbox("Please choose an option",
        [SIDEBAR_OPTION_1, SIDEBAR_OPTION_2, SIDEBAR_OPTION_3])
    if app_mode == SIDEBAR_OPTION_1:
        # st.sidebar.success('To continue, select "Run the app"')
        st.title(MAIN_TITLE)
        show_brief_data()
    elif app_mode == SIDEBAR_OPTION_2:
        # readme_text.empty()
        run_app()
    elif app_mode == SIDEBAR_OPTION_3:
        # readme_text.empty()
        st.code(get_file_content_as_string(APP_FILE_NAME))

def show_brief_data():
    train_images, train_labels, test_images, test_labels, class_names = load_data()
    col1, col2 = st.beta_columns((1, 1))

    # First column
    with col1:
        with st.beta_expander("Show the first data."):
            show_data(train_images)

    # Second column
    with col2:
        with st.beta_expander("Show the labels."):
            show_data_labels(train_images, train_labels, class_names)

def run_app():
    st.title("Fashion MNIST data classification")
    SELECTED_OPTIMIZER = optimizer_selector_ui()
    SELECTED_METRIC = metric_selector_ui()
    SELECTED_EPOCH = epoch_selector_ui()
    if st.sidebar.button("Start classification"):
        run_mnist(selected_optimizer=SELECTED_OPTIMIZER, selected_metric=SELECTED_METRIC,
                  selected_epochs=SELECTED_EPOCH)
        selections = f'{SELECTED_OPTIMIZER} {SELECTED_METRIC} {SELECTED_EPOCH}'
        if selections not in HYPER_PARAMS:
            HYPER_PARAMS.append(selections)

    # Check if there are selected hyper parameters
    if HYPER_PARAMS: # Not empty list
        selections = st.sidebar.multiselect("Choose the results and compare (Optimizer Metric Epoch)", HYPER_PARAMS)
        if selections:
            st.header("Comparison")
            compare_plots(selections)
            # for selection in selections:
            #     if selection in SAVE_IMAGES:
            #         compare_plot(selection)

# Download a single file and make its content available as a string.
@st.cache(show_spinner=False) # Default is True to show a spinner when there is a cache miss.
def get_file_content_as_string(path):
    url = 'https://raw.githubusercontent.com/taeokimeng/Streamlit-MNIST/master/' + path
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

def compare_plots(hyper_parameters):
    for key in hyper_parameters:
        if key in SAVE_IMAGES:
            parameters = key.split(' ')
            description = f'Optimizer: **{parameters[0]}**, Metric: **{parameters[1]}**, Epoch: **{parameters[2]}**'
            st.write(description)
            col1, col2 = st.beta_columns((1, 1))
            with st.spinner("Loading..."):
                with col1:
                    st.pyplot(SAVE_IMAGES[key][0])

                with col2:
                    st.pyplot(SAVE_IMAGES[key][1])

# This is made for debug, and not used now
def compare_plot(hyper_parameters):
    col1, col2 = st.beta_columns((1, 1))
    print(hyper_parameters)
    # if hyper_parameters in SAVE_IMAGES:
    parameters = hyper_parameters.split(' ')
    print(parameters)
    description = f'Optimizer: {parameters[0]}, Metric: {parameters[1]}, Epoch: {parameters[2]}'
    print(description)
    st.subheader(description)
    #     # st.write(description)
    #     with st.spinner("Loading..."):
    with col1:
        st.pyplot(SAVE_IMAGES[hyper_parameters][0])

if __name__ == "__main__":
    main()