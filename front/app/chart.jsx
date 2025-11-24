import { LineChart, CartesianGrid, XAxis, YAxis, Tooltip, Legend, Line } from 'recharts';

export default function Chart({data}) {
    return (
        <LineChart data={data} margin={{ top: 10, right: 24, left: 0, bottom: 10 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
            <XAxis dataKey="index" stroke="#4b5563" tick={{ fill: '#4b5563' }} />
            <YAxis stroke="#4b5563" tick={{ fill: '#4b5563' }} />
            <Tooltip contentStyle={{ backgroundColor: '#ffffff', border: '1px solid #e5e7eb', color: '#111827' }} />
            <Legend wrapperStyle={{ color: '#6b7280' }} />
            <Line type="monotone" dataKey="value" stroke="#2563eb" strokeWidth={2} activeDot={{ r: 6 }} />
        </LineChart>
    );
}