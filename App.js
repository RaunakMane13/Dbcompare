import './App.css';
import { BrowserRouter as Router, Route, Routes, Link } from 'react-router-dom';
import HomePage from './HomePage';
import DemoPage from './DemoPage';
import SignupPage from './SIgnupPage';
import LoginPage from './LoginPage';
import ExplorePage from './Explore';
// import DatasetDetails from './Details';
// import Details from './Details';
import ProjectOverview from './ProjectOverview';
import FeaturesPage from './Features';
import EditRecord from './Update';
import ShardingPage from './QO';

export default function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<HomePage />} />
        <Route path="/demo" element={<DemoPage />} />
        <Route path="/signup" element={<SignupPage />} />
        <Route path="/login" element={<LoginPage />} />
        <Route path="/explore" element={<ExplorePage />} />
        {/* <Route path="/details/:filename" element={<Details />} />
        <Route path="/dataset/:datasetName" element={<DatasetDetails />} />
        <Route path="/uploads/:fileName" element={<DatasetDetails />} /> */}
        <Route path="/project-overview" element={<ProjectOverview />} />
        <Route path="/features" element={<FeaturesPage />} />
        <Route path="/updates" element={<EditRecord />} />
        <Route path="/sharding" element={<ShardingPage />} />
      </Routes>
    </Router>
  );
}