import streamlit as st
import os
import imageio
import tensorflow as tf
from src.utils import load_data, num_to_char
from src.model_util import load_saved_model

st.set_page_config(layout='wide')

with st.sidebar:
    st.image('https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png')
    st.title('LipCipher')
    st.info('A real-time lip-reading demo powered by deep learning.')

st.title('LipCipher App')

options = os.listdir(os.path.join('data', 's1'))
selected_video = st.selectbox('Choose a video', options)

col1, col2 = st.columns(2)

if options:
    with col1:
        st.info('Input Video (MP4 converted)')
        input_path = os.path.join('data', 's1', selected_video)
        output_path = 'temp_video.mp4'
        cmd = f'ffmpeg -i {input_path} -vcodec libx264 {output_path} -y'
        status = os.system(cmd)
        if status == 0 and os.path.exists(output_path):
            video = open(output_path, 'rb')
            st.video(video.read())
        else:
            st.error(f'Error converting video. FFmpeg status: {status}')

    with col2:
        st.info('Processed Frames (Model Input)')
        video, alignments = load_data(tf.convert_to_tensor(input_path))
        imageio.mimsave('animation.gif', video, fps=10)
        st.image('animation.gif', width=400)

        st.info('Model Output Tokens')
        model = load_saved_model('model')
        yhat = model.predict(tf.expand_dims(video, axis=0))
        decoded = tf.keras.backend.ctc_decode(yhat, [75], greedy=True)[0][0].numpy()
        st.text(decoded)

        st.info('Decoded Text')
        decoded_text = tf.strings.reduce_join(num_to_char(decoded)).numpy().decode('utf-8')
        st.text(decoded_text)