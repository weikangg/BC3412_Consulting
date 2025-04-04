import './App.css'
import ScoreTable from './components/ScoreTable'
import RiskTable from './components/RiskTable';

function App() {
  return (
    <div className="App">
      <h1>TransitionPath Insights Dashboard</h1>
      <ScoreTable />
      <RiskTable />
    </div>
  );
}

export default App
