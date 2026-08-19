// Runs real, pre-trained scikit-learn Random Forest models inside a plain
// Node Netlify Function. No Python runtime, no numpy — we trained the
// forests offline (see ml/train_models.py) and exported each tree as a
// plain nested JSON object; this file just walks those trees.
//
// This is standard practice for shipping sklearn models into JS-only
// environments: the MODEL is real (real splits, real learned thresholds,
// real class probabilities from real training data) — only the *runtime*
// is reimplemented, not the model itself.

function walkTree(node, features) {
  if (node.leaf) return node.proba;
  const value = features[node.feature] ?? 0;
  return value <= node.threshold ? walkTree(node.left, features) : walkTree(node.right, features);
}

// Averages the leaf class-probabilities across every tree in the forest —
// this is exactly what sklearn's RandomForestClassifier.predict_proba does.
function forestPredictProba(forestJson, features) {
  const nClasses = forestJson.trees[0] ? walkTree(forestJson.trees[0], features).length : 2;
  const totals = new Array(nClasses).fill(0);
  for (const tree of forestJson.trees) {
    const proba = walkTree(tree, features);
    proba.forEach((p, i) => { totals[i] += p; });
  }
  return totals.map(t => t / forestJson.trees.length);
}

// Builds the same one-hot + numeric + binary feature vector the Python
// training script built, from a single customer record (plain object with
// the same column names as the CSV).
function buildFeatureVector(record, modelMeta) {
  const features = {};

  for (const col of modelMeta.numeric_features) {
    const raw = record[col];
    features[col] = raw === undefined || raw === null || raw === "" ? 0 : Number(raw);
  }

  for (const col of modelMeta.binary_features) {
    features[col] = String(record[col]).trim().toLowerCase() === "yes" ? 1 : 0;
  }

  for (const col of modelMeta.categorical_features) {
    const categories = modelMeta.categorical_categories[col] || [];
    const rawVal = String(record[col] ?? "");
    for (const cat of categories) {
      // sklearn's OneHotEncoder feature naming convention: "<col>_<category>"
      features[`${col}_${cat}`] = rawVal === cat ? 1 : 0;
    }
  }

  return features;
}

function scoreChurn(record, churnModelJson) {
  const features = buildFeatureVector(record, churnModelJson);
  const proba = forestPredictProba(churnModelJson.model, features);
  // classes_ are ["0", "1"] where 1 = churn risk = "Yes"
  const churnIndex = churnModelJson.model.classes.indexOf("1");
  return {
    churnProbability: Math.round(proba[churnIndex] * 1000) / 10, // e.g. 63.4 (%)
    topDrivers: Object.entries(churnModelJson.model.feature_importances).slice(0, 5),
  };
}

function scoreUpsell(record, upsellModelJson) {
  const features = buildFeatureVector(record, upsellModelJson);
  const results = {};
  for (const [product, forest] of Object.entries(upsellModelJson.models)) {
    const proba = forestPredictProba(forest, features);
    const positiveIndex = forest.classes.indexOf("1");
    results[product] = {
      probability: Math.round(proba[positiveIndex] * 1000) / 10,
      topDrivers: Object.entries(forest.feature_importances).slice(0, 4),
    };
  }
  return results;
}

export { buildFeatureVector, forestPredictProba, scoreChurn, scoreUpsell };
