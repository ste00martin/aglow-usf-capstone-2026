/**
 * feed.tsx — Short-form content feed (TikTok-style vertical pager)
 *
 * UX flow:
 *   - Full-screen image cards, one per scroll snap
 *   - Scroll past an item → logs 'view' (≥2s) or 'skip' (<2s)
 *   - Tap the flag icon → report modal with categories
 *   - Report → logs 'report' with category (reward = -5.0)
 *
 * Data:
 *   - Uses FEED_ITEMS from data/feedData.ts (dummy picsum.photos images)
 *   - Replace with output of scripts/prepare_kaggle_feed.py for real Kaggle data
 *
 * Offline RL data collection:
 *   - Every interaction is appended to interaction_log.jsonl on device
 *   - Export via Share sheet (tap the export button) → feed to offline_rl_pipeline.py
 */

import {
  FlatList,
  View,
  Image,
  Text,
  Pressable,
  Modal,
  StyleSheet,
  Share,
  Alert,
  ViewToken,
  ActivityIndicator,
} from 'react-native';
import { useRef, useState, useCallback, useEffect } from 'react';
import { Ionicons } from '@expo/vector-icons';
import * as Haptics from 'expo-haptics';
import { useSafeAreaInsets } from 'react-native-safe-area-context';

import { FEED_ITEMS, REPORT_CATEGORIES, FeedItem, ReportCategory } from '../../data/feedData';
import { useInteractionLog } from '../../hooks/useInteractionLog';

// ── Report Modal ──────────────────────────────────────────────────────────────

type ReportModalProps = {
  visible: boolean;
  item: FeedItem | null;
  onSubmit: (category: ReportCategory) => void;
  onDismiss: () => void;
};

function ReportModal({ visible, item, onSubmit, onDismiss }: ReportModalProps) {
  return (
    <Modal
      visible={visible}
      transparent
      animationType="slide"
      onRequestClose={onDismiss}
    >
      <Pressable style={styles.modalBackdrop} onPress={onDismiss}>
        <Pressable style={styles.modalSheet} onPress={() => {}}>
          <View style={styles.modalHandle} />
          <Text style={styles.modalTitle}>Report this content</Text>
          <Text style={styles.modalSubtitle}>
            Reports help improve our content moderation model.
          </Text>

          {REPORT_CATEGORIES.map((cat) => (
            <Pressable
              key={cat}
              style={styles.categoryRow}
              onPress={() => onSubmit(cat)}
            >
              <Ionicons name="flag-outline" size={18} color="#e74c3c" />
              <Text style={styles.categoryText}>{cat}</Text>
            </Pressable>
          ))}

          <Pressable style={styles.cancelButton} onPress={onDismiss}>
            <Text style={styles.cancelText}>Cancel</Text>
          </Pressable>
        </Pressable>
      </Pressable>
    </Modal>
  );
}

// ── Feed Card ─────────────────────────────────────────────────────────────────

type FeedCardProps = {
  item: FeedItem;
  cardHeight: number;
  onReportPress: (item: FeedItem) => void;
};

function FeedCard({ item, cardHeight, onReportPress }: FeedCardProps) {
  const [loaded, setLoaded] = useState(false);

  return (
    <View style={[styles.card, { height: cardHeight }]}>
      {/* Full-screen image */}
      <View style={[StyleSheet.absoluteFill, styles.imageWrapper]}>
        <Image
          source={typeof item.uri === 'number' ? item.uri : { uri: item.uri }}
          style={{ width: '100%', height: '100%' }}
          resizeMode="contain"
          onLoad={() => setLoaded(true)}
        />
      </View>

      {!loaded && (
        <View style={styles.loadingOverlay}>
          <ActivityIndicator color="#fff" size="large" />
        </View>
      )}

      {/* Dark gradient overlay (simulated with a View) */}
      <View style={styles.gradientOverlay} />

      {/* Bottom metadata */}
      <View style={styles.bottomInfo}>
        <Text style={styles.caption}>{item.caption}</Text>
        <View style={styles.tagsRow}>
          {item.tags.map((tag) => (
            <Text key={tag} style={styles.tag}>#{tag}</Text>
          ))}
        </View>
      </View>

      {/* Right sidebar actions */}
      <View style={styles.sidebar}>
        <Pressable
          style={styles.sidebarButton}
          onPress={() => onReportPress(item)}
          hitSlop={12}
        >
          <Ionicons name="flag-outline" size={28} color="#fff" />
          <Text style={styles.sidebarLabel}>Report</Text>
        </Pressable>

        {/* Source badge */}
        <View style={styles.sourceBadge}>
          <Text style={styles.sourceBadgeText}>
            {item.source === 'kaggle' ? 'Kaggle' : 'Demo'}
          </Text>
        </View>
      </View>
    </View>
  );
}

// ── Main Screen ───────────────────────────────────────────────────────────────

export default function FeedScreen() {
  const insets = useSafeAreaInsets();
  const [CARD_HEIGHT, setCardHeight] = useState(0);

  const { startView, getViewDurationMs, logInteraction, getLogPath, getLogs } = useInteractionLog();

  const [reportTarget, setReportTarget] = useState<FeedItem | null>(null);

  // Track current visible item for view-duration logging
  const currentItemRef = useRef<FeedItem | null>(null);
  const reportedIdsRef = useRef<Set<string>>(new Set());

  const getContentUri = useCallback((item: FeedItem): string => (
    item.assetPath ?? item.kaggleId ?? String(item.uri)
  ), []);

  const flushCurrentItem = useCallback(async (restartView: boolean): Promise<void> => {
    const currentItem = currentItemRef.current;
    if (!currentItem) return;
    const viewDurationMs = getViewDurationMs();
    if (reportedIdsRef.current.has(currentItem.id)) {
      if (restartView) {
        startView();
      }
      return;
    }

    await logInteraction(currentItem.id, getContentUri(currentItem), 'skip', undefined, viewDurationMs);

    if (restartView) {
      startView();
    }
  }, [getContentUri, getViewDurationMs, logInteraction, startView]);

  useEffect(() => {
    return () => {
      void flushCurrentItem(false);
    };
  }, [flushCurrentItem]);

  const handleViewableItemsChanged = useRef(
    ({ viewableItems }: { viewableItems: ViewToken[] }) => {
      const newItem = viewableItems[0]?.item as FeedItem | undefined;
      if (!newItem) return;

      // Log the item we just scrolled away from
      if (currentItemRef.current && currentItemRef.current.id !== newItem.id) {
        void flushCurrentItem(false);
      }

      currentItemRef.current = newItem;
      startView();
    }
  ).current;

  const viewabilityConfig = useRef({
    itemVisiblePercentThreshold: 60,
  }).current;

  const handleReportPress = useCallback((item: FeedItem) => {
    Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium);
    setReportTarget(item);
  }, []);

  const handleReportSubmit = useCallback(async (category: ReportCategory) => {
    if (!reportTarget) return;

    await logInteraction(reportTarget.id, getContentUri(reportTarget), 'report', category);
    reportedIdsRef.current.add(reportTarget.id);
    setReportTarget(null);

    Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
    Alert.alert(
      'Report submitted',
      'Thanks — this helps fine-tune the content model.',
      [{ text: 'OK' }]
    );
  }, [getContentUri, logInteraction, reportTarget]);

  const handleExportLogs = useCallback(async () => {
    await flushCurrentItem(true);
    const logs = await getLogs();
    if (logs.length === 0) {
      Alert.alert('No data yet', 'Scroll through some content first to generate interaction data.');
      return;
    }
    const summary = [
      `Total interactions: ${logs.length}`,
      `Reports: ${logs.filter(l => l.action === 'report').length}`,
      `Long views (≥2s): ${logs.filter(l => l.action === 'view').length}`,
      `Skips: ${logs.filter(l => l.action === 'skip').length}`,
      `Log file: ${getLogPath()}`,
    ].join('\n');

    Alert.alert('Interaction Log', summary, [
      {
        text: 'Share log file',
        onPress: () => Share.share({ url: getLogPath(), message: 'interaction_log.jsonl' }),
      },
      { text: 'Close', style: 'cancel' },
    ]);
  }, [flushCurrentItem, getLogs, getLogPath]);

  const renderItem = useCallback(({ item }: { item: FeedItem }) => (
    <FeedCard
      item={item}
      cardHeight={CARD_HEIGHT}
      onReportPress={handleReportPress}
    />
  ), [CARD_HEIGHT, handleReportPress]);

  const getItemLayout = useCallback((_: unknown, index: number) => ({
    length: CARD_HEIGHT,
    offset: CARD_HEIGHT * index,
    index,
  }), [CARD_HEIGHT]);

  return (
    <View
      style={{ flex: 1, backgroundColor: '#000' }}
      onLayout={(e) => setCardHeight(e.nativeEvent.layout.height)}
    >
      {CARD_HEIGHT > 0 && (
        <FlatList
          data={FEED_ITEMS}
          keyExtractor={(item) => item.id}
          renderItem={renderItem}
          pagingEnabled
          showsVerticalScrollIndicator={false}
          snapToInterval={CARD_HEIGHT}
          decelerationRate="fast"
          onViewableItemsChanged={handleViewableItemsChanged}
          viewabilityConfig={viewabilityConfig}
          getItemLayout={getItemLayout}
          initialNumToRender={2}
          maxToRenderPerBatch={3}
          windowSize={5}
        />
      )}

      {/* Export button — top right */}
      <Pressable
        style={[styles.exportButton, { top: insets.top + 8 }]}
        onPress={handleExportLogs}
        hitSlop={8}
      >
        <Ionicons name="download-outline" size={22} color="#fff" />
      </Pressable>

      <ReportModal
        visible={reportTarget !== null}
        item={reportTarget}
        onSubmit={handleReportSubmit}
        onDismiss={() => setReportTarget(null)}
      />
    </View>
  );
}

// ── Styles ────────────────────────────────────────────────────────────────────
const styles = StyleSheet.create({
  card: {
    width: '100%',
    backgroundColor: '#000',
    overflow: 'hidden',
  },
  imageWrapper: {
    alignItems: 'center',
    justifyContent: 'center',
  },
  loadingOverlay: {
    ...StyleSheet.absoluteFillObject,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#111',
  },
  gradientOverlay: {
    ...StyleSheet.absoluteFillObject,
    // Simulate bottom gradient: transparent top → dark bottom
    backgroundColor: 'transparent',
    // React Native doesn't natively support gradients without expo-linear-gradient,
    // so we use a semi-transparent overlay anchored to the bottom via absolute positioning
  },
  // Dark scrim just over the bottom text area
  bottomInfo: {
    position: 'absolute',
    bottom: 90,
    left: 16,
    right: 72,
    backgroundColor: 'rgba(0,0,0,0.45)',
    borderRadius: 10,
    padding: 10,
  },
  caption: {
    color: '#fff',
    fontSize: 15,
    fontWeight: '600',
    lineHeight: 21,
  },
  tagsRow: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    marginTop: 4,
    gap: 6,
  },
  tag: {
    color: '#7ecfff',
    fontSize: 13,
  },
  sidebar: {
    position: 'absolute',
    right: 12,
    bottom: 100,
    alignItems: 'center',
    gap: 20,
  },
  sidebarButton: {
    alignItems: 'center',
    gap: 4,
  },
  sidebarLabel: {
    color: '#fff',
    fontSize: 11,
  },
  sourceBadge: {
    backgroundColor: 'rgba(255,255,255,0.15)',
    borderRadius: 6,
    paddingHorizontal: 7,
    paddingVertical: 3,
  },
  sourceBadgeText: {
    color: '#ccc',
    fontSize: 10,
  },
  exportButton: {
    position: 'absolute',
    right: 14,
    backgroundColor: 'rgba(0,0,0,0.5)',
    borderRadius: 20,
    padding: 8,
  },
  // ── Report Modal ──
  modalBackdrop: {
    flex: 1,
    backgroundColor: 'rgba(0,0,0,0.6)',
    justifyContent: 'flex-end',
  },
  modalSheet: {
    backgroundColor: '#1e1e1e',
    borderTopLeftRadius: 20,
    borderTopRightRadius: 20,
    paddingHorizontal: 20,
    paddingBottom: 36,
    paddingTop: 12,
  },
  modalHandle: {
    width: 40,
    height: 4,
    backgroundColor: '#555',
    borderRadius: 2,
    alignSelf: 'center',
    marginBottom: 16,
  },
  modalTitle: {
    color: '#fff',
    fontSize: 18,
    fontWeight: '700',
    marginBottom: 4,
  },
  modalSubtitle: {
    color: '#888',
    fontSize: 13,
    marginBottom: 16,
  },
  categoryRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 12,
    paddingVertical: 14,
    borderBottomWidth: StyleSheet.hairlineWidth,
    borderColor: '#333',
  },
  categoryText: {
    color: '#f0f0f0',
    fontSize: 15,
  },
  cancelButton: {
    marginTop: 16,
    alignItems: 'center',
    paddingVertical: 12,
    backgroundColor: '#2d2d2d',
    borderRadius: 10,
  },
  cancelText: {
    color: '#aaa',
    fontSize: 15,
  },
});
