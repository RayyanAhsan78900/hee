import streamlit as st

# Streamlit app title
st.title("Basic Python Script with Streamlit")

# Welcome message
st.write("Welcome to the Basic Python Script!")

# Get user input
name = st.text_input("What's your name?")

if name:
    # Personalized greeting
    st.write(f"Hello, {name}! Let's do some basic math.")

    # Get two numbers from the user
    num1 = st.number_input("Enter the first number:", value=0.0, step=0.1)
    num2 = st.number_input("Enter the second number:", value=0.0, step=0.1)

    # Perform basic arithmetic operations
    sum_result = num1 + num2
    diff_result = num1 - num2
    product_result = num1 * num2
    quotient_result = num1 / num2 if num2 != 0 else "undefined (cannot divide by zero)"

    # Display the results
    st.subheader("Results:")
    st.write(f"{num1} + {num2} = {sum_result}")
    st.write(f"{num1} - {num2} = {diff_result}")
    st.write(f"{num1} * {num2} = {product_result}")
    st.write(f"{num1} / {num2} = {quotient_result}")

    # Farewell message
    st.write("\nThank you for using the Basic Python Script. Goodbye!")
