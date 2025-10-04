import React from 'react';
import { TrendingUp, TrendingDown, Minus, Brain, Target, Activity, Zap } from 'lucide-react';

const SentimentDashboard = ({ sentimentData }) => {
  // Default data if none provided
  const data = sentimentData || {
    overall: 'Bullish',
    score: 75,
    confidence: 87,
    trend: 'up',
    fearGreed: 68,
    predictions: [
      { coin: 'BTC', sentiment: 'positive', change: 2.4, confidence: 87 },
      { coin: 'ETH', sentiment: 'positive', change: 1.8, confidence: 79 },
      { coin: 'ADA', sentiment: 'negative', change: -0.9, confidence: 72 },
      { coin: 'SOL', sentiment: 'positive', change: 3.2, confidence: 84 }
    ]
  };

  return (
    <div className="bg-gray-800/50 rounded-2xl p-6 border border-gray-700">
      {/* Dashboard Header */}
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center space-x-3">
          <div className="bg-green-500 p-2 rounded-lg">
            <Brain className="w-5 h-5 text-white" />
          </div>
          <div>
            <h2 className="text-xl font-bold text-white">Sentiment Dashboard</h2>
            <p className="text-sm text-gray-400">Real-time market mood analysis</p>
          </div>
        </div>
        <div className="flex items-center space-x-2 bg-green-500/20 px-3 py-1 rounded-lg">
          <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>
          <span className="text-green-400 text-sm font-semibold">LIVE</span>
        </div>
      </div>

      {/* Main Sentiment Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
        {/* Overall Sentiment */}
        <div className="bg-gray-900 rounded-xl p-4 border border-green-500/30">
          <div className="flex items-center justify-between mb-2">
            <span className="text-gray-400 text-sm">Overall Mood</span>
            {data.trend === 'up' ? (
              <TrendingUp className="w-4 h-4 text-green-400" />
            ) : data.trend === 'down' ? (
              <TrendingDown className="w-4 h-4 text-red-400" />
            ) : (
              <Minus className="w-4 h-4 text-yellow-400" />
            )}
          </div>
          <div className="text-2xl font-bold text-green-400 mb-1">{data.overall}</div>
          <div className="w-full bg-gray-700 rounded-full h-2">
            <div 
              className="bg-green-500 h-2 rounded-full transition-all duration-1000"
              style={{ width: `${data.score}%` }}
            ></div>
          </div>
          <div className="text-xs text-gray-400 mt-1">Sentiment Score: {data.score}/100</div>
        </div>

        {/* Confidence Meter */}
        <div className="bg-gray-900 rounded-xl p-4 border border-blue-500/30">
          <div className="flex items-center justify-between mb-2">
            <span className="text-gray-400 text-sm">AI Confidence</span>
            <Target className="w-4 h-4 text-blue-400" />
          </div>
          <div className="text-2xl font-bold text-blue-400 mb-1">{data.confidence}%</div>
          <div className="w-full bg-gray-700 rounded-full h-2">
            <div 
              className="bg-gradient-to-r from-blue-500 to-purple-500 h-2 rounded-full transition-all duration-1000"
              style={{ width: `${data.confidence}%` }}
            ></div>
          </div>
          <div className="text-xs text-gray-400 mt-1">Prediction Accuracy</div>
        </div>

        {/* Fear & Greed */}
        <div className="bg-gray-900 rounded-xl p-4 border border-yellow-500/30">
          <div className="flex items-center justify-between mb-2">
            <span className="text-gray-400 text-sm">Fear & Greed</span>
            <Zap className="w-4 h-4 text-yellow-400" />
          </div>
          <div className="text-2xl font-bold text-yellow-400 mb-1">{data.fearGreed}/100</div>
          <div className="w-full bg-gray-700 rounded-full h-2">
            <div 
              className="bg-gradient-to-r from-red-500 via-yellow-500 to-green-500 h-2 rounded-full transition-all duration-1000"
              style={{ width: `${data.fearGreed}%` }}
            ></div>
          </div>
          <div className="text-xs text-gray-400 mt-1">Market Sentiment Index</div>
        </div>
      </div>

      {/* Coin Predictions Grid */}
      <div className="mb-6">
        <h3 className="text-lg font-semibold text-white mb-4">Coin Predictions</h3>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          {data.predictions.map((coin, index) => (
            <div key={index} className="bg-gray-900 rounded-lg p-3 border border-gray-600">
              <div className="flex items-center justify-between mb-2">
                <span className="text-white font-semibold text-sm">{coin.coin}</span>
                <div className={`text-xs px-2 py-1 rounded ${
                  coin.sentiment === 'positive' ? 'bg-green-500/20 text-green-400' :
                  coin.sentiment === 'negative' ? 'bg-red-500/20 text-red-400' :
                  'bg-yellow-500/20 text-yellow-400'
                }`}>
                  {coin.sentiment}
                </div>
              </div>
              <div className={`text-lg font-bold ${
                coin.change >= 0 ? 'text-green-400' : 'text-red-400'
              }`}>
                {coin.change >= 0 ? '+' : ''}{coin.change}%
              </div>
              <div className="flex items-center justify-between text-xs text-gray-400">
                <span>Confidence</span>
                <span>{coin.confidence}%</span>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Sentiment Timeline (Simple) */}
      <div>
        <h3 className="text-lg font-semibold text-white mb-4">Sentiment Timeline</h3>
        <div className="bg-gray-900 rounded-xl p-4">
          <div className="flex items-center justify-between text-sm text-gray-400 mb-2">
            <span>Last 24 hours</span>
            <span>Now</span>
          </div>
          <div className="flex items-end space-x-1 h-20">
            {[65, 72, 58, 81, 75, 68, 79, 75].map((value, index) => (
              <div key={index} className="flex-1 flex flex-col items-center">
                <div 
                  className="w-3 bg-gradient-to-t from-green-400 to-green-600 rounded-t transition-all duration-500"
                  style={{ height: `${value}%` }}
                ></div>
                <div className="text-xs text-gray-500 mt-1">{index}h</div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};

export default SentimentDashboard;