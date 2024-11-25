import React, { useState, useEffect, useRef } from 'react';
import './App.css';

const App = () => {
  const [mode, setMode] = useState('training'); // 'training' or 'testing'
  const [language, setLanguage] = useState('asl'); // 'asl' or 'isl'
  const [currentGesture, setCurrentGesture] = useState(null);
  const [feedback, setFeedback] = useState('');
  const videoRef = useRef(null);

  useEffect(() => {
    requestGesture();
    startVideoStream();
  }, [language]);

  const startVideoStream = () => {
    navigator.mediaDevices
      .getUserMedia({ video: true })
      .then((stream) => {
        videoRef.current.srcObject = stream;
      })
      .catch((err) => {
        console.error('Error accessing webcam: ', err);
      });
  };

  const requestGesture = () => {
    const gestures = language === 'asl' ? ['A', 'B', 'C'] : ['X', 'Y', 'Z'];
    const randomGesture = gestures[Math.floor(Math.random() * gestures.length)];
    setCurrentGesture(randomGesture);
  };

  return (
    <div className="App">
      <div className="header">
        <h1>Sign Language Detector</h1>
      </div>

      <div className="language-toggle">
        <button className={language === 'asl' ? 'active' : ''} onClick={() => setLanguage('asl')}>ASL</button>
        <button className={language === 'isl' ? 'active' : ''} onClick={() => setLanguage('isl')}>ISL</button>
      </div>

      <div className="content-platform">
        <div className="content">
          <div className="gesture-display">
            {mode === 'training' ? (
              <div className="gesture-info">
                <h2>Gesture: {currentGesture}</h2>
                <img src={`path_to_image_folder/${currentGesture}.jpg`} alt={`Hand sign for ${currentGesture}`} />
              </div>
            ) : (
              <div className="gesture-info">
                <h2>Make Gesture for Alphabet: {currentGesture}</h2>
              </div>
            )}
          </div>

          <div className="camera-feed">
            <video ref={videoRef} autoPlay></video>
            <button className="check-gesture-button" onClick={() => setFeedback('Checking...')}>Check Gesture</button>
          </div>
        </div>
      </div>

      <div className="mode-toggle">
        <button className={mode === 'training' ? 'active' : ''} onClick={() => setMode('training')}>Training Mode</button>
        <button className={mode === 'testing' ? 'active' : ''} onClick={() => setMode('testing')}>Testing Mode</button>
      </div>

      <div className="feedback">
        <h2>{feedback}</h2>
      </div>
    </div>
  );
};

export default App;
