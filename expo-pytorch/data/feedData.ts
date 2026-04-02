/**
 * feedData.ts
 *
 * Type definitions and the default checked-in Flickr8k starter feed.
 * The starter dataset lives in kaggleFeedItems.ts so the app boots cleanly on
 * fresh clones. For a larger local-only dataset, generate the "full" profile:
 *
 *   python scripts/prepare_kaggle_feed.py --profile full --num-images 500
 *
 * That writes kaggleFeedItems.local.ts and assets/feed-local/, both gitignored.
 * If you want to use that larger dataset locally, temporarily swap the import
 * below from ./kaggleFeedItems to ./kaggleFeedItems.local.
 */

import { KAGGLE_FEED_ITEMS } from './kaggleFeedItems';

export type FeedItem = {
  id: string;
  uri: string | number;  // number = require() bundled asset reference
  assetPath?: string;    // repo-relative image path for exported interaction logs / training
  caption: string;
  tags: string[];
  source: 'kaggle' | 'dummy';
  kaggleId?: string;
};

export const FEED_ITEMS: FeedItem[] = KAGGLE_FEED_ITEMS;

export const REPORT_CATEGORIES = [
  'Inappropriate / NSFW',
  'Violence / Gore',
  'Spam or Misleading',
  'Hate Speech',
  'Irrelevant Content',
  'Other',
] as const;

export type ReportCategory = typeof REPORT_CATEGORIES[number];
