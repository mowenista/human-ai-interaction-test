console.log("Script is running");

const sendBtn = document.querySelector('#send-btn');
const promptInput = document.querySelector('#prompt-input');
const chatWindow = document.querySelector('#chat-window');

console.log("sendBtn:", sendBtn);
console.log("promptInput:", promptInput);
console.log("chatWindow:", chatWindow);

// Enable send button based on input
promptInput.addEventListener('input', function(event) {
    sendBtn.disabled = event.target.value ? false : true;
});


// Send message function
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
        // Log the raw response for debugging
        console.log("Raw response object:", response);
    
        // Check for any non-OK status codes
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status} - ${response.statusText}`);
        }
    
        return response.json();
    })
    .then(data => {
        // Append the bot's response to the chat window
        const botMessage = document.createElement('div');
        botMessage.className = 'd-flex';
        botMessage.innerHTML = `
            <img src="/static/aiphoto.webp" class="rounded-pill mx-1" width="70" alt="AI Profile Image">
            <div class="chat-bubble chat-bubble-ai">${data.response}</div>`;
        chatWindow.appendChild(botMessage);

        // Create a container for the Vega-Lite chart
        if (data.vega_spec) {
            const chartContainer = document.createElement('div');
            chartContainer.id = `vis-${Date.now()}`; // Unique ID for each chart
            chartContainer.style.marginTop = '10px'; // Add some spacing
            chatWindow.appendChild(chartContainer); // Append the chart container below the AI response

            // Render the Vega-Lite chart inside the container
            vegaEmbed(`#${chartContainer.id}`, data.vega_spec)
                .catch(error => {
                    console.error("Error rendering Vega-Lite chart:", error);
                });
        }

        // Scroll to the bottom of the chat window
        chatWindow.scrollTop = chatWindow.scrollHeight;
    })
    .catch(error => {
        // Log detailed fetch error
        console.error("Fetch error encountered:", error);
    
        // Optionally, display an error message in the UI
        const errorMessage = document.createElement('div');
        errorMessage.className = 'alert alert-danger';
        errorMessage.textContent = `An error occurred: ${error.message}`;
        document.body.appendChild(errorMessage);
    });
}

// Scroll to the bottom of the chat window after new content is added
chatWindow.scrollTop = chatWindow.scrollHeight;


// Send message on Enter key press
promptInput.addEventListener('keyup', function(event) {
    if (event.keyCode === 13) {
        sendBtn.click();
    }
});

// Send button click event
sendBtn.addEventListener('click', sendMessage);


// CSV Upload functionality
console.log("Script is running");

const fileUploadArea = document.getElementById('file-upload-area');
const fileInput = document.getElementById('file-input');
const uploadError = document.getElementById('upload-error');
const uploadSuccess = document.getElementById('upload-success');
const tablePreview = document.getElementById('table-preview');
const tableHead = document.getElementById('table-head');
const tableBody = document.getElementById('table-body');

// Drag-and-drop functionality
fileUploadArea.addEventListener('dragover', (event) => {
    event.preventDefault();
    fileUploadArea.classList.add('border-primary');
});

fileUploadArea.addEventListener('dragleave', () => {
    fileUploadArea.classList.remove('border-primary');
});

fileUploadArea.addEventListener('drop', (event) => {
    event.preventDefault();
    fileUploadArea.classList.remove('border-primary');
    const files = event.dataTransfer.files;
    handleFileUpload(files[0]);
});

// Click to upload functionality
fileUploadArea.addEventListener('click', () => {
    fileInput.click();
});

fileInput.addEventListener('change', (event) => {
    const file = event.target.files[0];
    handleFileUpload(file);
});

// File upload function
function handleFileUpload(file) {
    if (!file || file.type !== 'text/csv') {
        console.error("Invalid file type. Please upload a CSV file.");
        showError("Invalid file type. Please upload a CSV file.");
        return;
    }

    const formData = new FormData();
    formData.append('file', file);

    console.log("Uploading file:", file.name);

    fetch('/upload-csv', {
        method: 'POST',
        body: formData
    })
    .then(response => {
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        return response.json();
    })
    .then(data => {
        console.log("File uploaded successfully:", data);
        showSuccess(data.message);  // Display the message from the server with the file name
        
        // Read the uploaded file and parse it using d3-dsv
        const reader = new FileReader();
        reader.onload = function(event) {
            const csvData = event.target.result;
            parseAndDisplayCSV(csvData);
        };
        reader.readAsText(file);
    })
    .catch(error => {
        console.error("Error during file upload:", error);
        showError("An error occurred while uploading the file.");
    });
}

// Function to parse and display the CSV data using d3-dsv
function parseAndDisplayCSV(csvData) {
    const parsedData = d3.csvParse(csvData, d3.autoType);
    console.log("Parsed CSV Data:", parsedData);
    
    // Show the table preview
    tablePreview.classList.remove('d-none');

    // Clear previous table data
    tableHead.innerHTML = '';
    tableBody.innerHTML = '';

    // Display table headers
    const headers = Object.keys(parsedData[0]);
    const headerRow = document.createElement('tr');
    headers.forEach(header => {
        const th = document.createElement('th');
        th.textContent = header;
        headerRow.appendChild(th);
    });
    tableHead.appendChild(headerRow);

    // Display first 5-10 rows of the parsed data
    parsedData.slice(0, 5).forEach(row => {
        const tr = document.createElement('tr');
        headers.forEach(header => {
            const td = document.createElement('td');
            td.textContent = row[header];
            tr.appendChild(td);
        });
        tableBody.appendChild(tr);
    });
}

// Show success message
function showSuccess(message) {
    uploadSuccess.textContent = message;
    uploadSuccess.classList.remove('d-none');
    uploadError.classList.add('d-none');
}

// Show error message
function showError(message) {
    uploadError.textContent = message;
    uploadError.classList.remove('d-none');
    uploadSuccess.classList.add('d-none');
}

// Show/Hide Table

const togglePreviewBtn = document.getElementById('toggle-preview-btn');

// Add event listener for the toggle button
togglePreviewBtn.addEventListener('click', () => {
    const tablePreview = document.getElementById('table-preview');
    if (tablePreview.classList.contains('d-none')) {
        tablePreview.classList.remove('d-none');
        togglePreviewBtn.textContent = 'Hide Table Preview';
    } else {
        tablePreview.classList.add('d-none');
        togglePreviewBtn.textContent = 'Show Table Preview';
    }
});
