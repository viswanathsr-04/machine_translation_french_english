import streamlit as st
from transformers import MarianMTModel, MarianTokenizer

# Load pre-trained model and tokenizer
model_name = "Helsinki-NLP/opus-mt-fr-en"  # French to English translation model
model = MarianMTModel.from_pretrained(model_name)
tokenizer = MarianTokenizer.from_pretrained(model_name)


# Streamlit app
def main():
    st.title("French to English Translation App")

    # Input text area
    input_text = st.text_area("Enter text in French", "")

    # Translate button
    if st.button("Translate"):
        if input_text.strip() == "":
            st.warning("Please enter some text to translate.")
        else:
            # Tokenize input text
            input_text_tokens = tokenizer(input_text, return_tensors="pt")

            # Translate text
            translated_tokens = model.generate(**input_text_tokens)

            # Decode translated text
            translated_text = tokenizer.decode(
                translated_tokens[0], skip_special_tokens=True
            )

            # Display translated text
            st.success("Translated text:")
            st.write(translated_text)


if __name__ == "__main__":
    main()
