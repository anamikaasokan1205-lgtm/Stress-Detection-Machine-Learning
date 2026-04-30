import streamlit as st
import pickle

# Load model and vectorizer
model = pickle.load(open("model.sav", "rb"))
vectorizer = pickle.load(open("vectorizer.sav", "rb"))

st.title("Stress Detection Chatbot")

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.last_result = None  # store last stress prediction
    st.session_state.messages.append({"role": "assistant", "content": "Hello! How are you feeling today?"})

# Display previous messages
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Chat input
if user_input := st.chat_input("Type your message..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.chat_message("user").write(user_input)

    # If last_result was stress and user says yes/no, handle that
    if st.session_state.last_result == 1 and "yes" in user_input.lower():
        response = (
            "Here’s a helpful resource: "
            "[Stress Management Techniques](https://www.helpguide.org/mental-health/stress/stress-management). "
            "It has practical strategies you can try."
        )
    elif st.session_state.last_result == 1 and "no" in user_input.lower():
        response = (
            "That’s completely fine. Even small steps like taking a walk, "
            "listening to music, or talking to a friend can help reduce stress."
        )
    else:
        # Guardrail: check input length before classification
        if len(user_input.split()) < 3:
            response = "Could you tell me a bit more about how you're feeling?"
        else:
            # Classify normally
            X = vectorizer.transform([user_input])
            result = model.predict(X)[0]
            st.session_state.last_result = result  # save prediction

            if result == 0:
                response = "Glad to hear that! It seems you're not showing signs of stress. Keep taking care of yourself."
            else:
                response = (
                    "It looks like you may be stressed. "
                    "Try taking a short break, practicing deep breathing, or writing down your thoughts. "
                    "Would you like me to direct you to a stress management website?"
                )

    st.session_state.messages.append({"role": "assistant", "content": response})
    st.chat_message("assistant").write(response)