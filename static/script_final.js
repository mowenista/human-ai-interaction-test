console.log("Script is running");

const sendBtn = document.querySelector('#send-btn');
const promptInput = document.querySelector('#prompt-input');
const chatWindow = document.querySelector('#chat-window');

console.log("sendBtn:", sendBtn);
console.log("promptInput:", promptInput);
console.log("chatWindow:", chatWindow);

// Your existing code...
promptInput.addEventListener('input', function(event) {
    sendBtn.disabled = event.target.value ? false : true;
});

function sendMessage() {
    const prompt = promptInput.value;
    if (!prompt) {
        console.log("Prompt is empty, returning.");
        return;
    }

    // Clear the input field
    promptInput.value = '';

    // Append user's message to the chat window
    const userMessage = document.createElement('div');
    userMessage.className = 'd-flex justify-content-end';
    userMessage.innerHTML = `<div class="chat-bubble chat-bubble-user">${prompt}</div>`;
    chatWindow.appendChild(userMessage);

    // Scroll to the bottom of the chat window (in case of overflow)
    chatWindow.scrollTop = chatWindow.scrollHeight;

    console.log("User message appended: ", prompt);

    fetch('/query', {
        method: 'POST',
        body: JSON.stringify({ prompt }),
        headers: {
            'Content-Type': 'application/json'
        }
    })
    .then(response => {
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        return response.json();
    })
    .then(data => {
        console.log("Response from server: ", data.response);
        
        // Append the bot's response to the chat window
        const botMessage = document.createElement('div');
        botMessage.className = 'd-flex';
        botMessage.innerHTML = `
            <img src="/static/aiphoto.webp" class="rounded-pill mx-1" width="70" alt="AI Profile Image">
            <div class="chat-bubble chat-bubble-ai">${data.response}</div>`;
        chatWindow.appendChild(botMessage);

        // Scroll to the bottom of the chat window
        chatWindow.scrollTop = chatWindow.scrollHeight;
    })
    .catch(error => {
        console.error("Error during fetch:", error);
    });
}

promptInput.addEventListener('keyup', function(event) {
    if (event.keyCode === 13) {
        sendBtn.click();
    }
});

sendBtn.addEventListener('click', sendMessage);

