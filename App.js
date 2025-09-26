import './App.css';
import { BrowserRouter as Router, Route, Routes, Link } from 'react-router-dom';
import HomePage from './HomePage';
import DemoPage from './DemoPage';
import VisualizePage from './Visualize';
import Sharding from './Sharding';
import Editor from './Editor';
import SignupPage from './SignupPage';
import LoginPage from './LoginPage';
import ExplorePage from './Explore';
// import DatasetDetails from './Details';
// import Details from './Details';
import ProjectOverview from './ProjectOverview';
// import FeaturesPage from './Features';
// import EditRecord from './Update';
// import ShardingPage from './QO';

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
        <Route path="/visualize" element={<VisualizePage />} />
        <Route path="/editor" element={<Editor />} />
        <Route path="/sharding" element={<Sharding />} />
      </Routes>
    </Router>
  );
}