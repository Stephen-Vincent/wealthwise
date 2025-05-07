export default function AIPortfolioSummary() {
  return (
    <div className="mb-8">
      <h3 className="text-xl font-bold mb-4">AI Portfolio Summary</h3>
      <div className="space-y-4 text-sm text-gray-800">
        <div>
          <h4 className="font-bold">Portfolio Overview:</h4>
          <p>
            Your portfolio achieved a total return of +200%, driven by steady
            market growth while minimizing major risks during downturns.
          </p>
        </div>
        <div>
          <h4 className="font-bold">Key Events Impacting Your Portfolio:</h4>
          <ul className="list-disc list-inside space-y-1">
            <li>📈 2010–2015: Tech boom led by mobile and cloud innovation.</li>
            <li>📉 2018: U.S.-China tariffs caused short-term losses.</li>
            <li>
              📈 2020–2021: Post-COVID recovery led by tech and healthcare
              sectors.
            </li>
          </ul>
        </div>
        <div>
          <h4 className="font-bold">Risk and Resilience:</h4>
          <p>
            Moderate volatility with quick recovery during market dips. A
            well-balanced mix of tech and consumer staples supported stability.
          </p>
        </div>
        <div>
          <h4 className="font-bold">Final Insight:</h4>
          <p>
            Diversification and long-term investing helped you grow capital
            efficiently while managing risk exposure.
          </p>
        </div>
      </div>
    </div>
  );
}
