# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Step 3: Fetches weather using Gemini by using a langchain agent

    Usage: 
        pip install -r requirements.txt
        streamlit run <filename>
"""


import streamlit as st
import vertexai
from vertexai.generative_models import GenerativeModel, Part
from vertexai.preview import reasoning_engines

import base64
import json
from enum import Enum
import dotenv
import os

dotenv.load_dotenv()

# init vertex ai
vertexai.init(
    project=os.environ.get("PROJECT_ID"), 
    location=os.environ.get("REGION"))

# init gemini 1.5 flash model
model = GenerativeModel("gemini-1.5-flash-001")

# create a langchain agent
agent = reasoning_engines.LangchainAgent(
    model="gemini-1.5-flash-001")


# function to generate image completions from gemini
def get_image_info(image_content):
    global model

    prompt = """What is this place and where is it located?
    Output the name, description, and location with city, state, country."""

    generation_config = {
        "max_output_tokens": 8192,
        "temperature": 1,
        "top_p": 0.95,
        "response_mime_type": "application/json"
    }

    # get image as Part
    image_part = Part.from_data(
        mime_type="image/jpeg",
        data=image_content
    )

    # invoke the model with the part object
    response = model.generate_content(
        [image_part, prompt],
        generation_config=generation_config
    )

    return json.loads(response.candidates[0].text)


def display_identify_button():
    # show button to identify image
    upfile = st.session_state["image_file"]
    if upfile is None: 
        return

    # display button only if image_info is empty
    if st.button("Identify"):
        # get image content as base64
        file_content = upfile.read()
        # convert to base64
        image_content = base64.b64encode(file_content).decode('utf-8')
        # invoke gemini
        if st.session_state["image_info"] is None:
            with st.spinner("Identifying..."):
                st.session_state["image_info"] = get_image_info(image_content)


def display_image_upload():
    # upload image
    st.subheader("Upload an image")
    upfile = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"], 
                              label_visibility="hidden")
    
    if upfile is None and st.session_state["image_file"] is not None:
        upfile = st.session_state["image_file"]
    elif upfile is not None:
        st.session_state["image_file"] = upfile
        
        # reset session when a new image is uploaded
        st.session_state["image_content"] = None
        st.session_state["image_info"] = None
        st.session_state["chat_history"] = []

    # if file is not empty, then display it
    if upfile is not None:
        st.image(upfile, width=250)


# display the generated response
def display_image_info():
    data = st.session_state["image_info"]
    if data is None:
        return

    st.header(data["name"], divider="rainbow")
    st.write(data["description"])
    st.subheader("Location")
    st.text(", ".join([
        data["location"]["city"], 
        data["location"]["state"], 
        data["location"]["country"]]))

   
# display the weather
def display_weather():
    data = st.session_state["image_info"]
    if data is None:
        return

    st.subheader("Weather")
    with st.spinner("Fetching weather..."):
        st.write(get_weather_response(data["location"]["city"]))


# get weather response from gemini using the agent
def get_weather_response(city):
    global model
    global agent

    prompt = f"""What is the weather in {city}?
    Output the temperature in celcius and the climate."""

    return agent.query(input=prompt)["output"]


# main streamlit app
def main():
    # set session state
    if "image_file" not in st.session_state:
        st.session_state["image_file"] = None
    if "image_content" not in st.session_state:
        st.session_state["image_content"] = None
    if "image_info" not in st.session_state:
        st.session_state["image_info"] = None
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    st.title("Travel Companion")

    display_image_upload()
    display_identify_button() 
    display_image_info()
    display_weather()


# run the main app
if __name__ == "__main__":
    main()
