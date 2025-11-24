'use client'; // This line makes it a client-side component

import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

// Sample data for the graph
const data = [
  { name: 'Jan', Sales: 4000, Profit: 2400 },
  { name: 'Feb', Sales: 3000, Profit: 1398 },
  { name: 'Mar', Sales: 2000, Profit: 9800 },
  { name: 'Apr', Sales: 2780, Profit: 3908 },
  { name: 'May', Sales: 1890, Profit: 4800 },
  { name: 'Jun', Sales: 2390, Profit: 3800 },
  { name: 'Jul', Sales: 3490, Profit: 4300 },
];

// Example Next.js Client Component function
async function runPrediction(inputData) {
  const response = await fetch('/api/python-call', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(inputData),
  });

  if (!response.ok) {
    throw new Error('Prediction API failed');
  }

  const result = await response.json();
  return result.prediction;
}

export default function Home() {
  return (
    <main style={{ padding: '20px' }}>
      <h1>My Basic Next.js Graph</h1>
      {/* ResponsiveContainer ensures the chart fills its parent container */}
      <div style={{ width: '100%', height: 300 }}>
        <ResponsiveContainer width="100%" height="100%">
          <LineChart
            data={data}
            margin={{
              top: 5,
              right: 30,
              left: 20,
              bottom: 5,
            }}
          >
            <CartesianGrid strokeDasharray="3 3" />
            {/* XAxis takes the 'name' property from the data */}
            <XAxis dataKey="name" />
            <YAxis />
            <Tooltip />
            <Legend />
            {/* Line 1 plots 'Sales' */}
            <Line type="monotone" dataKey="Sales" stroke="#8884d8" activeDot={{ r: 8 }} />
            {/* Line 2 plots 'Profit' */}
            <Line type="monotone" dataKey="Profit" stroke="#82ca9d" />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </main>
  );
}
