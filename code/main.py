from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
from pycoingecko import CoinGeckoAPI
import time
from datetime import datetime, timedelta

# 1) Carga tus artefactos
model_rf      = joblib.load('rf_model.pkl')
scaler_feats  = joblib.load('scaler_X.pkl')
scaler_target = joblib.load('scaler_y.pkl')

feature_cols = [
    'current_price',
    'total_volume',
    'ath',
    'atl',
    'price_change_percentage_24h',
    'ath_change_percentage',
    'atl_change_percentage'
]

# 2) Instancia CoinGecko
cg_api = CoinGeckoAPI()

# Cache para evitar muchas llamadas a la API
recommendations_cache = {}
cache_timestamp = None
CACHE_DURATION = 300  # 5 minutos

# 3) Funciones auxiliares
def lookup_crypto_id(symbol: str) -> str:
    res = cg_api.search(query=symbol)
    for coin in res.get('coins', []):
        if coin['symbol'].lower() == symbol.lower() or coin['name'].lower() == symbol.lower():
            return coin['id']
    if res.get('coins'):
        return res['coins'][0]['id']
    raise ValueError(f"No se encontró ninguna moneda para '{symbol}'")

def get_crypto_features(crypto_id: str) -> dict:
    data = cg_api.get_coins_markets(
        vs_currency='usd',
        ids=[crypto_id],
        price_change_percentage='24h'
    )
    if not data:
        raise ValueError(f"No se encontró '{crypto_id}' en CoinGecko")
    e = data[0]
    return {
        'current_price'               : e['current_price'],
        'total_volume'                : e['total_volume'],
        'ath'                         : e['ath'],
        'atl'                         : e['atl'],
        'market_cap'                  : e.get('market_cap', 0),
        'price_change_percentage_24h' : e.get('price_change_percentage_24h', 0.0),
        'ath_change_percentage'       : e.get('ath_change_percentage', 0.0),
        'atl_change_percentage'       : e.get('atl_change_percentage', 0.0),
        'last_updated'                : e.get('last_updated', ''),
        'homepage'                    : e.get('homepage', [''])[0],
        'image'                       : e.get('image'),
    }

def get_price_history(crypto_id: str):
    chart = cg_api.get_coin_market_chart_by_id(id=crypto_id, vs_currency='usd', days=7)
    return chart['prices']

def rf_predict_and_categorize(values_list):
    arr         = np.array(values_list).reshape(1, -1)
    arr_scaled  = scaler_feats.transform(arr)
    pred_scaled = model_rf.predict(arr_scaled).reshape(-1, 1)
    pred        = scaler_target.inverse_transform(pred_scaled)[0, 0]
    if   pred > 10: category = "ALTA_OPORTUNIDAD"
    elif pred >  5: category = "MODERADA_OPORTUNIDAD"
    elif pred >  0: category = "BAJA_OPORTUNIDAD"
    else:           category = "NO_RECOMENDADO"
    return round(pred, 2), category

# 4) Sistema de Recomendaciones
def get_top_cryptos(limit=50):
    """Obtiene las top cryptos por market cap"""
    return cg_api.get_coins_markets(
        vs_currency='usd',
        order='market_cap_desc',
        per_page=limit,
        page=1,
        sparkline=False,
        price_change_percentage='24h'
    )

def analyze_crypto_for_recommendations(crypto_data):
    """Analiza una crypto y devuelve su predicción y score"""
    try:
        values = [
            crypto_data['current_price'],
            crypto_data['total_volume'],
            crypto_data['ath'],
            crypto_data['atl'],
            crypto_data.get('price_change_percentage_24h', 0.0),
            crypto_data.get('ath_change_percentage', 0.0),
            crypto_data.get('atl_change_percentage', 0.0)
        ]
        
        prediction, category = rf_predict_and_categorize(values)
        
        # Score adicional basado en métricas técnicas
        volume_score = min(crypto_data['total_volume'] / 1e9, 5)  # Normalizado a 5
        market_cap_score = min(crypto_data.get('market_cap', 0) / 1e10, 5)  # Normalizado a 5
        price_momentum = crypto_data.get('price_change_percentage_24h', 0)
        
        # Score compuesto
        technical_score = (volume_score + market_cap_score + (price_momentum/10)) / 3
        final_score = (prediction + technical_score) / 2
        
        return {
            'id': crypto_data['id'],
            'symbol': crypto_data['symbol'],
            'name': crypto_data['name'],
            'image': crypto_data['image'],
            'current_price': crypto_data['current_price'],
            'market_cap': crypto_data.get('market_cap', 0),
            'total_volume': crypto_data['total_volume'],
            'price_change_24h': crypto_data.get('price_change_percentage_24h', 0),
            'prediction': prediction,
            'category': category,
            'final_score': round(final_score, 2),
            'risk_level': get_risk_level(crypto_data),
            'recommendation_reason': get_recommendation_reason(prediction, category, crypto_data)
        }
    except Exception as e:
        return None

def get_risk_level(crypto_data):
    """Determina el nivel de riesgo basado en volatilidad y market cap - Versión más balanceada"""
    market_cap = crypto_data.get('market_cap', 0)
    price_change = abs(crypto_data.get('price_change_percentage_24h', 0))
    
    # Ajustar los umbrales para que más cryptos sean clasificadas como bajo/medio riesgo
    if market_cap > 100e9:  # > 100B (muy grandes)
        if price_change < 8:
            return "BAJO"
        elif price_change < 20:
            return "MEDIO"
        else:
            return "ALTO"
    elif market_cap > 10e9:  # 10B - 100B (grandes)
        if price_change < 12:
            return "BAJO"
        elif price_change < 25:
            return "MEDIO"
        else:
            return "ALTO"
    elif market_cap > 1e9:  # 1B - 10B (medianas)
        if price_change < 15:
            return "MEDIO"
        else:
            return "ALTO"
    else:  # < 1B (pequeñas)
        if price_change < 10:
            return "MEDIO"
        else:
            return "ALTO"

def get_recommendation_reason(prediction, category, crypto_data):
    """Genera una razón personalizada para la recomendación"""
    price_change = crypto_data.get('price_change_percentage_24h', 0)
    market_cap = crypto_data.get('market_cap', 0)
    
    reasons = []
    
    if prediction > 10:
        reasons.append("Predicción de crecimiento muy alta")
    elif prediction > 5:
        reasons.append("Predicción de crecimiento positiva")
    elif prediction > 0:
        reasons.append("Predicción de crecimiento moderada")
    else:
        reasons.append("Predicción de crecimiento negativa")
    
    if price_change > 5:
        reasons.append("momentum positivo 24h")
    elif price_change < -5:
        reasons.append("corrección reciente (oportunidad de compra)")
    
    if market_cap > 50e9:
        reasons.append("activo establecido")
    elif market_cap > 5e9:
        reasons.append("capitalización media")
    else:
        reasons.append("alto potencial de crecimiento")
    
    return ", ".join(reasons)

def generate_recommendations(risk_tolerance="MEDIO", limit=10):
    """Genera recomendaciones personalizadas"""
    global recommendations_cache, cache_timestamp
    
    # Verificar cache
    current_time = time.time()
    if (cache_timestamp and 
        current_time - cache_timestamp < CACHE_DURATION and 
        recommendations_cache):
        cryptos_analyzed = recommendations_cache
    else:
        # Obtener y analizar cryptos
        top_cryptos = get_top_cryptos(50)
        cryptos_analyzed = []
        
        for crypto in top_cryptos:
            analysis = analyze_crypto_for_recommendations(crypto)
            if analysis:
                cryptos_analyzed.append(analysis)
        
        # Actualizar cache
        recommendations_cache = cryptos_analyzed
        cache_timestamp = current_time
    
    # Filtrar categorías no recomendadas primero
    valid_recommendations = [c for c in cryptos_analyzed if c['category'] != 'NO_RECOMENDADO']
    
    # Filtrar por tolerancia al riesgo de manera más flexible
    if risk_tolerance == "BAJO":
        # Solo riesgo bajo y medio
        filtered = [c for c in valid_recommendations if c['risk_level'] in ['BAJO', 'MEDIO']]
        # Si no hay suficientes, incluir algunos de riesgo alto con mejor score
        if len(filtered) < limit:
            high_risk_backup = [c for c in valid_recommendations if c['risk_level'] == 'ALTO']
            high_risk_backup.sort(key=lambda x: x['final_score'], reverse=True)
            filtered.extend(high_risk_backup[:limit - len(filtered)])
    
    elif risk_tolerance == "ALTO":
        # Incluir todos los niveles de riesgo
        filtered = valid_recommendations
    
    else:  # MEDIO
        # Priorizar bajo y medio, pero incluir alto si es necesario
        preferred = [c for c in valid_recommendations if c['risk_level'] in ['BAJO', 'MEDIO']]
        if len(preferred) < limit:
            # Agregar cryptos de alto riesgo con mejor score para completar
            high_risk = [c for c in valid_recommendations if c['risk_level'] == 'ALTO']
            high_risk.sort(key=lambda x: x['final_score'], reverse=True)
            preferred.extend(high_risk[:limit - len(preferred)])
        filtered = preferred
    
    # Ordenar por score final
    filtered.sort(key=lambda x: x['final_score'], reverse=True)
    
    return filtered[:limit]

def get_portfolio_suggestions(budget=1000, risk_tolerance="MEDIO"):
    """Genera sugerencias de portafolio diversificado"""
    recommendations = generate_recommendations(risk_tolerance, 20)
    
    if not recommendations:
        return []
    
    # Distribución según categorías
    portfolio = []
    remaining_budget = budget
    
    # Asignación por categorías
    alta_oportunidad = [r for r in recommendations if r['category'] == 'ALTA_OPORTUNIDAD'][:3]
    moderada_oportunidad = [r for r in recommendations if r['category'] == 'MODERADA_OPORTUNIDAD'][:4]
    baja_oportunidad = [r for r in recommendations if r['category'] == 'BAJA_OPORTUNIDAD'][:3]
    
    allocations = []
    
    # Alta oportunidad: 40% del presupuesto
    if alta_oportunidad:
        allocation_per_crypto = (budget * 0.4) / len(alta_oportunidad)
        for crypto in alta_oportunidad:
            amount = allocation_per_crypto / crypto['current_price']
            allocations.append({
                **crypto,
                'suggested_amount': round(amount, 6),
                'suggested_investment': round(allocation_per_crypto, 2),
                'allocation_percentage': round(40 / len(alta_oportunidad), 1)
            })
    
    # Moderada oportunidad: 35% del presupuesto
    if moderada_oportunidad:
        allocation_per_crypto = (budget * 0.35) / len(moderada_oportunidad)
        for crypto in moderada_oportunidad:
            amount = allocation_per_crypto / crypto['current_price']
            allocations.append({
                **crypto,
                'suggested_amount': round(amount, 6),
                'suggested_investment': round(allocation_per_crypto, 2),
                'allocation_percentage': round(35 / len(moderada_oportunidad), 1)
            })
    
    # Baja oportunidad: 25% del presupuesto (más estables)
    if baja_oportunidad:
        allocation_per_crypto = (budget * 0.25) / len(baja_oportunidad)
        for crypto in baja_oportunidad:
            amount = allocation_per_crypto / crypto['current_price']
            allocations.append({
                **crypto,
                'suggested_amount': round(amount, 6),
                'suggested_investment': round(allocation_per_crypto, 2),
                'allocation_percentage': round(25 / len(baja_oportunidad), 1)
            })
    
    return allocations

# 5) Configura Flask
app = Flask(__name__)
CORS(app)

@app.route('/', methods=['GET','POST'])
def home():
    html_form = """
<html>
  <head>
    <title>RF Crypto Predictor & Recommendations</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
      * { box-sizing: border-box; margin: 0; padding: 0; }
      body {
        font-family: Arial, sans-serif;
        background: #2e2e2e;
        color: #fff;
        padding: 2rem;
      }
      .recommendations-container {
        max-width: 1200px;
        margin: 40px auto;
        padding: 20px;
        background-color: #12161c;
        border-radius: 12px;
        box-shadow: 0 0 20px rgba(0,0,0,0.15);
      }
      h1, h2 {
        text-align: center;
        color: #00ffff;
        margin-bottom: 1.5rem;
      }
      .tabs {
        display: flex;
        justify-content: center;
        margin-bottom: 20px;
        gap: 10px;
      }
      .tab {
        padding: 10px 20px;
        background: #161b22;
        border: 1px solid #2c323c;
        border-radius: 6px;
        cursor: pointer;
        transition: all 0.3s;
      }
      .tab.active, .tab:hover {
        background: #00bcd4;
        color: #12161c;
      }
      .tab-content {
        display: none;
      }
      .tab-content.active {
        display: block;
      }
      .input-row {
        display: flex;
        gap: 12px;
        justify-content: center;
        margin-bottom: 20px;
        flex-wrap: wrap;
      }
      .input-row input, .input-row select {
        background: #0e1116;
        border: 1px solid #2c323c;
        border-radius: 6px;
        color: #fff;
        padding: 5px 10px;
        min-width: 120px;
      }
      button {
        position: relative;
        padding: 8px 16px;
        border: none;
        border-radius: 6px;
        background: #00bcd4;
        color: #12161c;
        font-size: 1rem;
        font-weight: 500;
        cursor: pointer;
        transition: background 0.2s;
      }
      button:hover {
        background: #26c6da;
      }
      .loader {
        display: none;
        position: absolute;
        top: 50%%;
        left: 50%%;
        width: 20px;
        height: 20px;
        margin: -10px 0 0 -10px;
        border: 3px solid #ccc;
        border-top: 3px solid #00bcd4;
        border-radius: 50%%;
        animation: spin 1s linear infinite;
      }
      @keyframes spin { 100%% { transform: rotate(360deg); } }
      hr {
        border: none;
        border-top: 1px solid #2c323c;
        margin: 20px 0;
      }
      .crypto-card {
        opacity: 0;
        transform: translateY(20px);
        transition: opacity 0.5s ease, transform 0.5s ease;
        background-color: #161b22;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #2c323c;
        margin-bottom: 20px;
      }
      .crypto-card.show {
        opacity: 1;
        transform: translateY(0);
      }
      .crypto-card h3 {
        display: flex;
        align-items: center;
        gap: 8px;
        color: #00ffff;
        margin-bottom: 10px;
      }
      .crypto-card img {
        border-radius: 4px;
      }
      .crypto-card .info p {
        margin: 6px 0;
        color: #d1d5db;
      }
      .recommendations-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
        gap: 20px;
        margin-top: 20px;
      }
      .recommendation-card {
        background: #161b22;
        border: 1px solid #2c323c;
        border-radius: 10px;
        padding: 15px;
        transition: transform 0.2s;
      }
      .recommendation-card:hover {
        transform: translateY(-2px);
        border-color: #00bcd4;
      }
      .risk-badge {
        display: inline-block;
        padding: 4px 8px;
        border-radius: 4px;
        font-size: 0.8em;
        font-weight: bold;
        margin-left: 10px;
      }
      .risk-BAJO { background: #4caf50; color: white; }
      .risk-MEDIO { background: #ff9800; color: white; }
      .risk-ALTO { background: #f44336; color: white; }
      .category-badge {
        display: inline-block;
        padding: 4px 8px;
        border-radius: 4px;
        font-size: 0.8em;
        font-weight: bold;
        margin: 5px 0;
      }
      .cat-ALTA_OPORTUNIDAD { background: #4caf50; color: white; }
      .cat-MODERADA_OPORTUNIDAD { background: #ff9800; color: white; }
      .cat-BAJA_OPORTUNIDAD { background: #2196f3; color: white; }
      canvas { width: 100%%; height: auto; margin-top: 20px; }
      .error { color: #f44336; }
      .portfolio-summary {
        background: #0e1116;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 20px;
      }
    </style>
  </head>
  <body>
    <div class="recommendations-container">
      <h1>RF Crypto Predictor & Recommendations</h1>
      
      <div class="tabs">
        <div class="tab active" onclick="showTab('predict')">Predicción Individual</div>
        <div class="tab" onclick="showTab('recommendations')">Recomendaciones</div>
        <div class="tab" onclick="showTab('portfolio')">Portafolio Sugerido</div>
      </div>

      <!-- Tab Predicción Individual -->
      <div id="predict" class="tab-content active">
        <form id="predict-form" method="post" action="/?tab=predict">
          <div class="input-row">
            <input id="symbol" name="symbol" placeholder="bitcoin" required>
            <button type="submit">
              <span id="btn-text">Predecir</span>
              <div class="loader" id="loader"></div>
            </button>
          </div>
        </form>
        %s
        <canvas id="price-chart"></canvas>
      </div>

      <!-- Tab Recomendaciones -->
      <div id="recommendations" class="tab-content">
        <form method="post" action="/?tab=recommendations">
          <div class="input-row">
            <select name="risk_tolerance">
              <option value="BAJO">Riesgo Bajo</option>
              <option value="MEDIO" selected>Riesgo Medio</option>
              <option value="ALTO">Riesgo Alto</option>
            </select>
            <input type="number" name="limit" placeholder="10" value="10" min="1" max="20">
            <button type="submit">Generar Recomendaciones</button>
          </div>
        </form>
        %s
      </div>

      <!-- Tab Portafolio -->
      <div id="portfolio" class="tab-content">
        <form method="post" action="/?tab=portfolio">
          <div class="input-row">
            <input type="number" name="budget" placeholder="1000" value="1000" min="100" step="100">
            <select name="risk_tolerance">
              <option value="BAJO">Riesgo Bajo</option>
              <option value="MEDIO" selected>Riesgo Medio</option>
              <option value="ALTO">Riesgo Alto</option>
            </select>
            <button type="submit">Generar Portafolio</button>
          </div>
        </form>
        %s
      </div>
    </div>

    <script>
      function showTab(tabName) {
        document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
        document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
        document.querySelector(`[onclick="showTab('${tabName}')"]`).classList.add('active');
        document.getElementById(tabName).classList.add('active');
        
        // Update URL hash without reload
        history.replaceState(null, null, `#${tabName}`);
      }

      // Handle tab from URL or form submission
      window.onload = () => {
        const urlParams = new URLSearchParams(window.location.search);
        const tab = urlParams.get('tab') || window.location.hash.slice(1) || 'predict';
        if (tab && ['predict', 'recommendations', 'portfolio'].includes(tab)) {
          showTab(tab);
        }

        // Chart handling
        const dataEl = document.getElementById('chart-data');
        if (dataEl) {
          const history = JSON.parse(dataEl.textContent);
          new Chart(document.getElementById('price-chart'), {
            type: 'line',
            data: {
              labels: history.map(p => new Date(p[0]).toLocaleDateString()),
              datasets: [{ data: history.map(p => p[1]), borderColor: '#00ffff', fill: false }]
            },
            options: {
              elements: { point: { radius: 0 } },
              plugins: { legend: { display: false } }
            }
          });
        }
      };

      document.getElementById('predict-form').onsubmit = () => {
        document.getElementById('btn-text').style.visibility = 'hidden';
        document.getElementById('loader').style.display = 'block';
      };
    </script>
  </body>
</html>
"""

    resultado_predict = ""
    resultado_recommendations = ""
    resultado_portfolio = ""
    
    if request.method == 'POST':
        tab = request.args.get('tab', 'predict')
        
        if tab == 'predict':
            sym = request.form['symbol'].strip()
            try:
                cid       = lookup_crypto_id(sym)
                feats     = get_crypto_features(cid)
                vals      = [feats[c] for c in feature_cols]
                pred, cat = rf_predict_and_categorize(vals)
                history   = get_price_history(cid)

                # Formateos y valores extra
                cp    = feats['current_price']
                mc    = f"{feats['market_cap']:,}"
                vol   = f"{feats['total_volume']:,}"
                chg   = feats['price_change_percentage_24h']
                color = 'green' if chg >= 0 else 'red'
                upd   = feats['last_updated'].replace('T',' ').replace('Z','')
                home  = feats['homepage']
                chart_script = f'<script id="chart-data" type="application/json">{history}</script>'

                # Consejo según predicción
                advice = (
                  "Gran oportunidad: predicción alta, podrías asignar posición moderada." if pred>10 else
                  "Oportunidad moderada: tendencia positiva, vigila volatilidad." if pred>5 else
                  "Riesgo bajo: crecimiento leve, mantén expectativas moderadas." if pred>0 else
                  "No recomendado: predicción negativa, mejor espera."
                )

                resultado_predict = f"""
<div class="crypto-card show">
  <h3>
    <img src="{feats['image']}" width="32" height="32" alt="{sym}">
    {sym.capitalize()} ({cid})
  </h3>
  <div class="info">
    <p><strong>Precio actual:</strong> ${cp}</p>
    <p><strong>Market Cap:</strong> ${mc}</p>
    <p><strong>Vol 24h:</strong> {vol}</p>
    <p><strong>Predicción:</strong> {pred}%</p>
    <p><strong>Categoría:</strong> {cat}</p>
    <p><strong>Cambio 24h:</strong> <span style="color:{color};">{chg:.2f}%</span></p>
    <p title="All Time High"><strong>ATH:</strong> ${feats['ath']}</p>
    <p title="All Time Low"><strong>ATL:</strong> ${feats['atl']}</p>
    <p><small>Última actualización: {upd}</small></p>
    {f'<p><a href="{home}" target="_blank" style="color:#00bcd4;">Sitio oficial</a></p>' if home else ''}
    <p><em>{advice}</em></p>
  </div>
</div>
{chart_script}
"""
            except Exception as e:
                resultado_predict = f"<p class='error'>Error: {e}</p>"
        
        elif tab == 'recommendations':
            try:
                risk_tolerance = request.form.get('risk_tolerance', 'MEDIO')
                limit = int(request.form.get('limit', 10))
                recommendations = generate_recommendations(risk_tolerance, limit)
                
                if recommendations:
                    cards = ""
                    for rec in recommendations:
                        cards += f"""
<div class="recommendation-card">
  <h4>
    <img src="{rec['image']}" width="24" height="24" alt="{rec['symbol']}">
    {rec['name']} ({rec['symbol'].upper()})
    <span class="risk-badge risk-{rec['risk_level']}">{rec['risk_level']}</span>
  </h4>
  <div class="category-badge cat-{rec['category']}">{rec['category']}</div>
  <p><strong>Precio:</strong> ${rec['current_price']}</p>
  <p><strong>Predicción:</strong> {rec['prediction']}%</p>
  <p><strong>Score Final:</strong> {rec['final_score']}</p>
  <p><strong>Cambio 24h:</strong> <span style="color:{'green' if rec['price_change_24h'] >= 0 else 'red'};">{rec['price_change_24h']:.2f}%</span></p>
  <p><strong>Market Cap:</strong> ${rec['market_cap']:,}</p>
  <p><small><em>{rec['recommendation_reason']}</em></small></p>
</div>
"""
                    resultado_recommendations = f"""
<h2>Top {len(recommendations)} Recomendaciones (Riesgo {risk_tolerance})</h2>
<div class="recommendations-grid">
{cards}
</div>
"""
                else:
                    resultado_recommendations = "<p>No se encontraron recomendaciones para los criterios seleccionados.</p>"
            except Exception as e:
                resultado_recommendations = f"<p class='error'>Error: {e}</p>"
        
        elif tab == 'portfolio':
            try:
                budget = float(request.form.get('budget', 1000))
                risk_tolerance = request.form.get('risk_tolerance', 'MEDIO')
                portfolio = get_portfolio_suggestions(budget, risk_tolerance)
                
                if portfolio:
                    total_investment = sum(p['suggested_investment'] for p in portfolio)
                    cards = ""
                    for p in portfolio:
                        cards += f"""
<div class="recommendation-card">
  <h4>
    <img src="{p['image']}" width="24" height="24" alt="{p['symbol']}">
    {p['name']} ({p['symbol'].upper()})
  </h4>
  <div class="category-badge cat-{p['category']}">{p['category']}</div>
  <p><strong>Inversión sugerida:</strong> ${p['suggested_investment']}</p>
  <p><strong>Cantidad:</strong> {p['suggested_amount']} {p['symbol'].upper()}</p>
  <p><strong>% del portafolio:</strong> {p['allocation_percentage']}%</p>
  <p><strong>Precio actual:</strong> ${p['current_price']}</p>
  <p><strong>Predicción:</strong> {p['prediction']}%</p>
  <p><strong>Riesgo:</strong> <span class="risk-badge risk-{p['risk_level']}">{p['risk_level']}</span></p>
</div>
"""
                    resultado_portfolio = f"""
<div class="portfolio-summary">
  <h2>Portafolio Sugerido (${budget:,.0f} - Riesgo {risk_tolerance})</h2>
  <p><strong>Total asignado:</strong> ${total_investment:,.2f}</p>
  <p><strong>Efectivo restante:</strong> ${budget - total_investment:,.2f}</p>
  <p><strong>Número de activos:</strong> {len(portfolio)}</p>
</div>
<div class="recommendations-grid">
{cards}
</div>
"""
                else:
                    resultado_portfolio = "<p>No se pudo generar un portafolio con los criterios seleccionados.</p>"
            except Exception as e:
                resultado_portfolio = f"<p class='error'>Error: {e}</p>"

    return html_form % (resultado_predict, resultado_recommendations, resultado_portfolio)

@app.route('/api/predict-crypto', methods=['POST'])
def predict_crypto_api():
    payload = request.json or {}
    sym     = payload.get('symbol','').strip()
    if not sym:
        return jsonify({'error':'Debes enviar {"symbol":"bitcoin"}'}), 400
    try:
        cid       = lookup_crypto_id(sym)
        feats     = get_crypto_features(cid)
        vals      = [feats[c] for c in feature_cols]
        pred, cat = rf_predict_and_categorize(vals)
        return jsonify({
            'symbol'    : sym.lower(),
            'crypto_id' : cid,
            'prediction': pred,
            'category'  : cat,
            'image'     : feats['image']
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/recommendations', methods=['GET'])
def get_recommendations_api():
    risk_tolerance = request.args.get('risk_tolerance', 'MEDIO')
    limit = int(request.args.get('limit', 10))
    try:
        recommendations = generate_recommendations(risk_tolerance, limit)
        return jsonify({
            'recommendations': recommendations,
            'count': len(recommendations),
            'risk_tolerance': risk_tolerance
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/portfolio', methods=['POST'])
def get_portfolio_api():
    payload = request.json or {}
    budget = payload.get('budget', 1000)
    risk_tolerance = payload.get('risk_tolerance', 'MEDIO')
    try:
        portfolio = get_portfolio_suggestions(budget, risk_tolerance)
        total_investment = sum(p['suggested_investment'] for p in portfolio)
        return jsonify({
            'portfolio': portfolio,
            'total_budget': budget,
            'total_investment': total_investment,
            'remaining_cash': budget - total_investment,
            'asset_count': len(portfolio),
            'risk_tolerance': risk_tolerance
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__=='__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)