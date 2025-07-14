import AsyncStorage from '@react-native-async-storage/async-storage';

export type CatEmbedding = {
  id: string;
  name: string;
  embedding: number[];
};

const STORAGE_KEY = 'cat_embeddings';

export async function getAllEmbeddings(): Promise<CatEmbedding[]> {
  const json = await AsyncStorage.getItem(STORAGE_KEY);
  if (!json) return [];
  return JSON.parse(json);
}

export async function saveEmbedding(cat: CatEmbedding) {
  const all = await getAllEmbeddings();
  all.push(cat);
  await AsyncStorage.setItem(STORAGE_KEY, JSON.stringify(all));
}

export function cosineSimilarity(a: number[], b: number[]): number {
  const dot = a.reduce((sum, ai, i) => sum + ai * b[i], 0);
  const normA = Math.sqrt(a.reduce((sum, ai) => sum + ai * ai, 0));
  const normB = Math.sqrt(b.reduce((sum, bi) => sum + bi * bi, 0));
  return dot / (normA * normB);
}

function euclideanDistance(a: number[], b: number[]): number {
  return Math.sqrt(a.reduce((sum, ai, i) => sum + (ai - b[i]) ** 2, 0));
}

export async function findBestMatch(embedding: number[], threshold = 0.15): Promise<(CatEmbedding & { score: number }) | null> {
  const all = await getAllEmbeddings();
  if (all.length === 0) return null;
  let best: CatEmbedding | null = null;
  let bestScore = Infinity;
  for (const cat of all) {
    const score = euclideanDistance(embedding, cat.embedding);
    if (score < bestScore) {
      best = cat;
      bestScore = score;
    }
  }
  // Always return the closest match and its score, even if similarity is low
  return { ...best!, score: bestScore };
} 