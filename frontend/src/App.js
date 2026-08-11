import { useState, useEffect } from 'react';
import axios from 'axios';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, Legend, ResponsiveContainer
} from 'recharts';

const API = 'http://localhost:8000';

function MetricsTable({ metrics }) {
  return (
    <div style={{ marginBottom: 32 }}>
      <h2>Model Comparison</h2>
      <table style={{ borderCollapse: 'collapse', width: '100%' }}>
        <thead>
          <tr style={{ background: '#f0f0f0' }}>
            {['Model', 'RMSE', 'MAE', 'PHM Score'].map(h => (
              <th key={h} style={{ padding: '8px 16px', border: '1px solid #ccc', textAlign: 'left' }}>{h}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {metrics.map(m => (
            <tr key={m.model}>
              <td style={{ padding: '8px 16px', border: '1px solid #ccc' }}>{m.model}</td>
              <td style={{ padding: '8px 16px', border: '1px solid #ccc' }}>{m.rmse}</td>
              <td style={{ padding: '8px 16px', border: '1px solid #ccc' }}>{m.mae}</td>
              <td style={{ padding: '8px 16px', border: '1px solid #ccc' }}>{m.phm}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function EngineChart({ engineData }) {
  if (!engineData) return null;

  const chartData = engineData.cycles.map((cycle, i) => ({
    cycle,
    true_rul:  engineData.true_rul_curve[i],
    lstm_pred: engineData.lstm_pred_curve[i],
  }));

  return (
    <div>
      <h2>Engine {engineData.engine_id} — RUL Prediction</h2>
      <p>
        True RUL at end: <strong>{engineData.true_rul}</strong> &nbsp;|&nbsp;
        RF Prediction: <strong>{engineData.rf_pred?.toFixed(1)}</strong> &nbsp;|&nbsp;
        LSTM Final: <strong>{engineData.lstm_final_pred?.toFixed(1)}</strong>
      </p>
      <ResponsiveContainer width="100%" height={350}>
        <LineChart data={chartData}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="cycle" label={{ value: 'Cycle', position: 'insideBottom', offset: -5 }} />
          <YAxis label={{ value: 'RUL', angle: -90, position: 'insideLeft' }} />
          <Tooltip />
          <Legend verticalAlign="top" />
          <Line type="monotone" dataKey="true_rul"  stroke="#2563eb" dot={false} name="True RUL" />
          <Line type="monotone" dataKey="lstm_pred" stroke="#dc2626" dot={false} name="LSTM Predicted" strokeDasharray="5 5" />
        </LineChart>
      </ResponsiveContainer>

      <h3 style={{ marginTop: 32 }}>Sensor Trajectories</h3>
      {Object.entries(engineData.sensors).slice(0, 4).map(([sensor, values]) => {
        const sensorData = engineData.cycles.map((cycle, i) => ({ cycle, value: values[i] }));
        return (
          <div key={sensor} style={{ marginBottom: 24 }}>
            <h4>{sensor}</h4>
            <ResponsiveContainer width="100%" height={150}>
              <LineChart data={sensorData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="cycle" />
                <YAxis />
                <Tooltip />
                <Line type="monotone" dataKey="value" stroke="#7c3aed" dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        );
      })}
    </div>
  );
}

export default function App() {
  const [metrics, setMetrics]       = useState([]);
  const [engineIds, setEngineIds]   = useState([]);
  const [selectedId, setSelectedId] = useState(1);
  const [engineData, setEngineData] = useState(null);
  const [loading, setLoading]       = useState(false);

  useEffect(() => {
    axios.get(`${API}/metrics`).then(r => setMetrics(r.data.results));
    axios.get(`${API}/engines`).then(r => setEngineIds(r.data.engine_ids));
  }, []);

  useEffect(() => {
    setLoading(true);
    axios.get(`${API}/engine/${selectedId}`)
      .then(r => { setEngineData(r.data); setLoading(false); })
      .catch(() => setLoading(false));
  }, [selectedId]);

  return (
    <div style={{ maxWidth: 900, margin: '0 auto', padding: 32, fontFamily: 'sans-serif' }}>
      <h1>Predictive Maintenance Dashboard</h1>
      <p style={{ color: '#666' }}>NASA C-MAPSS FD001 — Turbofan Engine RUL Prediction</p>

      <MetricsTable metrics={metrics} />

      <div style={{ marginBottom: 24 }}>
        <label><strong>Select Engine: </strong></label>
        <select
          value={selectedId}
          onChange={e => setSelectedId(Number(e.target.value))}
          style={{ padding: '4px 8px', fontSize: 16 }}
        >
          {engineIds.map(id => <option key={id} value={id}>{id}</option>)}
        </select>
      </div>

      {loading ? <p>Loading...</p> : <EngineChart engineData={engineData} />}
    </div>
  );
}
