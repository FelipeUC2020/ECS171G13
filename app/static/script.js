// Load available models on page load
console.log('Script loaded');
window.addEventListener('DOMContentLoaded', async function() {
    console.log('DOM loaded');
    const select = document.getElementById('model-select');
    console.log('Select element:', select);
    
    try {
        console.log('Fetching /models');
        const response = await fetch('/models');
        console.log('Response:', response);
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        const data = await response.json();
        console.log('Data:', data);
        // Clear existing options
        select.innerHTML = '';
        // Add fetched models
        data.models.forEach(model => {
            const option = document.createElement('option');
            option.value = model;
            option.textContent = model;
            select.appendChild(option);
        });
        console.log('Models loaded:', data.models);
    } catch (error) {
        console.error('Failed to load models:', error);
        // Keep default option
    }
});

document.getElementById('forecast-form').addEventListener('submit', async function(e) {
    e.preventDefault();
    
    const inputData = document.getElementById('input-data').value;
    const selectedModel = document.getElementById('model-select').value;
    let parsedData;
    
    try {
        parsedData = JSON.parse(inputData);
        if (!Array.isArray(parsedData) || parsedData.length !== 72 || !parsedData.every(hour => Array.isArray(hour) && hour.length === 4)) {
            throw new Error('Input must be a 72x4 array');
        }
    } catch (error) {
        // TODO: Improve error messages and user feedback
        document.getElementById('result').innerHTML = '<p style="color: red;">Invalid input format. Please enter a valid 72x4 JSON array.</p>';
        return;
    }
    
    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ input: parsedData, model: selectedModel }),
        });
        
        const result = await response.json();
        // TODO: Display predictions with a chart (e.g., using Chart.js)
        document.getElementById('result').innerHTML = '<h3>24-Hour Forecast:</h3><pre>' + JSON.stringify(result.prediction, null, 2) + '</pre>';
    } catch (error) {
        // TODO: Better error handling for network/server errors
        document.getElementById('result').innerHTML = '<p style="color: red;">Error: ' + error.message + '</p>';
    }
});