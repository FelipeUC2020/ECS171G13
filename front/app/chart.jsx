import { LineChart, CartesianGrid, XAxis, YAxis, Tooltip, Legend, Line } from 'recharts';

export default function Chart({ data, series }) {
    // series: optional array of { key, name, color, dot, activeDot }
    // If not provided, infer from data keys excluding 'index'
    const palette = ['#2563eb', '#10b981', '#ef4444', '#f59e0b', '#8b5cf6', '#06b6d4'];
    const inferredKeys = Array.isArray(data) && data.length > 0
        ? Object.keys(data[0]).filter(k => k !== 'index')
        : [];
    const seriesToPlot = Array.isArray(series) && series.length > 0
        ? series
        : inferredKeys.map((key, i) => ({ key, name: key, color: palette[i % palette.length] }));

    return (
        <LineChart data={data} margin={{ top: 10, right: 24, left: 0, bottom: 10 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
            <XAxis dataKey="index" stroke="#4b5563" tick={{ fill: '#4b5563' }} />
            <YAxis stroke="#4b5563" tick={{ fill: '#4b5563' }} />
            <Tooltip contentStyle={{ backgroundColor: '#ffffff', border: '1px solid #e5e7eb', color: '#111827' }} />
            <Legend wrapperStyle={{ color: '#6b7280' }} />
            {seriesToPlot.map((s, idx) => (
                <Line
                    key={s.key ?? idx}
                    type="monotone"
                    name={s.name ?? s.key}
                    dataKey={s.key}
                    stroke={s.color ?? palette[idx % palette.length]}
                    strokeWidth={2}
                    dot={s.dot ?? false}
                    activeDot={s.activeDot ?? { r: 4 }}
                />
            ))}
        </LineChart>
    );
}