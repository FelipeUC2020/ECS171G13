'use client';

import { ResponsiveContainer } from 'recharts';
import { useState, useEffect } from 'react';
import Chart from './chart';

export default function Home() {
  const [chartData, setChartData] = useState([]);
  const [inputData, setInputData] = useState([]);
  const [model, setModel] = useState('CNN');
  const [isLoading, setIsLoading] = useState(true);

  const runPrediction = async (inputData) => {
    setIsLoading(true);
    const response = await fetch('http://127.0.0.1:8000/run', { method: 'GET' });
    if (!response.ok) throw new Error('Prediction API failed');
    const result = await response.json();
    console.log("Received data:", result);

    const cnn = result.cnn_predictions[0];
    const lstm = result.lstm_predictions[0];
    const label = result.label;

    const len = Math.max(cnn.length, lstm.length, label.length);
    const combined_chart = Array.from({ length: len }, (_, i) => ({
      index: i + 1,
      CNN: cnn[i] ?? null,
      LSTM: lstm[i] ?? null,
      Real: label[i] ?? null,
    }));

    const combined_input = Array.from({ length: result.input.length }, (_, i) => ({
      index: i + 1,
      GlobalActivePower: result.input[i][0] ?? null,
      GlobalReactivePower: result.input[i][1] ?? null,
      Voltage: result.input[i][2] ?? null,
      GlobalIntensity: result.input[i][3] ?? null,
      SubMeter1: result.input[i][4] ?? null,
      SubMeter2: result.input[i][5] ?? null,
      SubMeter3: result.input[i][6] ?? null,
    }));

    setChartData(combined_chart);
    setInputData(combined_input);
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
            24 Hour Global Active Power Prediction
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

        <h1 className="text-2xl font-bold tracking-wide my-4">
          Input Data
        </h1>

        <div className="w-full h-96 p-3 bg-white rounded-lg border border-slate-200">
          { isLoading ? (
            <div className="flex items-center justify-center h-full">
              <p className="text-slate-500 animate-pulse">Loading prediction...</p>
            </div>
          ) : inputData && (
            <ResponsiveContainer width="100%" height="100%">
              <Chart data={inputData} />
            </ResponsiveContainer>
          )}
        </div>
      </div>
    </main>
  );
}