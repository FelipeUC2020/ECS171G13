'use client';

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

async function runPrediction(inputData) {
  const response = await fetch('http://127.0.0.1:8000/run', { method: 'GET' });
  if (!response.ok) throw new Error('Prediction API failed');
  const result = await response.json();
  return result.predictions; // or whatever shape you need
}

export default function Home() {
  return (
    <main style={{ padding: '20px' }}>
      <h1>My Basic Next.js Graph</h1>
      <div style={{ width: '100%', height: 300 }}>
        <ResponsiveContainer width="100%" height="100%">
          <LineChart
            data={data}
            margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
          >
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="name" />
            <YAxis />
            <Tooltip />
            <Legend />
            <Line type="monotone" dataKey="Sales" stroke="#8884d8" activeDot={{ r: 8 }} />
            <Line type="monotone" dataKey="Profit" stroke="#82ca9d" />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </main>
  );
}