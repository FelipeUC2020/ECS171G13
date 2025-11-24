'use client';

import { ResponsiveContainer } from 'recharts';
import { useState, useEffect } from 'react';
import Chart from './chart';

export default function Home() {
  const [chartData, setChartData] = useState([]);
  const [model, setModel] = useState('CNN');
  const [isLoading, setIsLoading] = useState(true);

  const runPrediction = async (inputData) => {
    setIsLoading(true);
    const response = await fetch('http://127.0.0.1:8000/run', { method: 'GET' });
    if (!response.ok) throw new Error('Prediction API failed');
    const result = await response.json();
    console.log("Received data:", result);
    console.log("Predictions:", result.predictions);
    console.log("LLM Recommendations:", result.llm_recommendations);
    const predictions = result.predictions[0].map((val, idx) => ({ index: idx + 1, value: val })); // or whatever shape you need
    setChartData(predictions);
    setIsLoading(false);
  }

  useEffect(() => {
    // Fetch prediction data and update chart
    const fetchData = async () => {
      await runPrediction();
    };
    fetchData();
  }, []);

  return (
    <main className="min-h-screen bg-gradient-to-b from-white to-slate-100 text-slate-800 p-10 flex items-center justify-center">
      <div className="w-full max-w-3xl p-6 bg-white border border-slate-200 rounded-xl shadow-lg">
        <div className="flex flex-row items-center justify-between">
          <h1 className="text-2xl font-bold tracking-wide mb-4">
            24 Hour Global Active Power Prediction ({model})
          </h1>

          <button 
            className="ml-auto px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition-colors cursor-pointer disabled:opacity-50"
            onClick={async () => {
              await runPrediction();
            }}
            disabled={isLoading}
          >
            New Sample
          </button>

        </div>
        <div className="w-full h-96 p-3 bg-white rounded-lg border border-slate-200">
          { isLoading ? (
            <div className="flex items-center justify-center h-full">
              <p className="text-slate-500 animate-pulse">Loading prediction...</p>
            </div>
          ) : chartData && (
            <ResponsiveContainer width="100%" height="100%">
              <Chart data={chartData} />
            </ResponsiveContainer>
          )}
        </div>
      </div>
    </main>
  );
}