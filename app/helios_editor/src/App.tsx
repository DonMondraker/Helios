import React, { useEffect, useRef, useState } from "react";
import { Streamlit } from "streamlit-component-lib";
import {
  Stage,
  Layer,
  Image as KonvaImage,
  Line,
  Circle,
  Text,
  Rect,
  Group,
} from "react-konva";
import useImage from "use-image";

import sampleImage from "./assets/battD.jpg";

type Tool = "select" | "draw" | "callout" | "detail" | "inset" | "focus";

type EditorLine = {
  id: string;
  name: string;
  x1: number;
  y1: number;
  x2: number;
  y2: number;
};

type EditorCallout = {
  id: string;
  name: string;
  label: string;
  circleX: number;
  circleY: number;
  anchorX: number;
  anchorY: number;
};

type EditorInset = {
  id: string;
  name: string;
  x: number;
  y: number;
  width: number;
  height: number;
  aspectRatio: number;
  sourceX: number;
  sourceY: number;
  sourceWidth: number;
  sourceHeight: number;
  showLeader: boolean;
  leaderAnchorX: number;
  leaderAnchorY: number;
};

type EditorInsetImage = {
  id: string;
  name: string;

  x: number;
  y: number;
  width: number;
  height: number;

  imageSrc: string;

  aspectRatio: number;

  showLeader: boolean;
  leaderAnchorX: number;
  leaderAnchorY: number;
};

type FocusObject = {
  id: string;
  name: string;
  polygon: { x: number; y: number }[];
  haloEnabled: boolean;
};

type HeliosEditorState = {
  version: 1;

  lines: EditorLine[];
  callouts: EditorCallout[];
  detailViews: EditorInset[];
  insetImages: EditorInsetImage[];
  focusObjects: FocusObject[];

  selectedObjectId: string | null;

  exportedImageDataUrl?: string;
  exportRequestId?: string;
};

type HeliosEditorProps = {
  imageSrc?: string;
  debug?: boolean;
  initialState?: Partial<HeliosEditorState>;
  focusObjectsFromStreamlit?: FocusObject[];
  aiSuggestions?: AiAnnotationSuggestions | null;
  pendingInsetAsset?: PendingInsetAsset | null;
  exportRequestId?: string | null;
};

type AiAnnotationSuggestions = {
  movement_lines?: {
    start: [number, number];
    end: [number, number];
  }[];
  callouts?: {
    label: string;
    circle: [number, number];
    end: [number, number];
  }[];
};

type PendingInsetAsset = {
  id: string;
  name: string;
  imageSrc: string;
};

const SNAP_ANGLES = [0, 45, 90, 135, 180, -45, -90, -135, -180];

const HALO_COLOR = "white";
const ANNOTATION_COLOR = "black";
const SELECTED_COLOR = "#50ae91";

function snapEndpoint(
  x1: number,
  y1: number,
  x2: number,
  y2: number
): { x: number; y: number } {
  const dx = x2 - x1;
  const dy = y2 - y1;
  const length = Math.hypot(dx, dy);

  if (length < 4) return { x: x2, y: y2 };

  const angle = Math.atan2(dy, dx) * (180 / Math.PI);

  const nearest = SNAP_ANGLES.reduce((best, current) => {
    const bestDiff = Math.abs(((angle - best + 180) % 360) - 180);
    const currentDiff = Math.abs(((angle - current + 180) % 360) - 180);
    return currentDiff < bestDiff ? current : best;
  });

  const radians = nearest * (Math.PI / 180);

  return {
    x: x1 + Math.cos(radians) * length,
    y: y1 + Math.sin(radians) * length,
  };
}

function InsetImageObject({
  inset,
  isSelected,
  tool,
  shiftDown,
  scale,
  setSelectedObjectId,
  setInsetImages,
}: {
  inset: EditorInsetImage;
  isSelected: boolean;
  tool: Tool;
  shiftDown: boolean;
  scale: number;
  setSelectedObjectId: React.Dispatch<React.SetStateAction<string | null>>;
  setInsetImages: React.Dispatch<React.SetStateAction<EditorInsetImage[]>>;
}) {
  const [img] = useImage(inset.imageSrc);
  const s = (value: number) => value / (scale > 0 ? scale : 1);

  const insetCenterX = inset.x + inset.width / 2;
  const insetCenterY = inset.y + inset.height / 2;

  return (
    <React.Fragment>
      {inset.showLeader && (
        <>
          <Line
            points={[inset.leaderAnchorX, inset.leaderAnchorY, insetCenterX, insetCenterY]}
            stroke={HALO_COLOR}
            strokeWidth={s(isSelected ? 10 : 8)}
            lineCap="round"
            lineJoin="round"
            listening={false}
          />
          <Line
            points={[inset.leaderAnchorX, inset.leaderAnchorY, insetCenterX, insetCenterY]}
            stroke={isSelected ? SELECTED_COLOR : ANNOTATION_COLOR}
            strokeWidth={s(isSelected ? 4 : 3)}
            lineCap="round"
            lineJoin="round"
          />
        </>
      )}

      <Group
        x={inset.x}
        y={inset.y}
        draggable={tool === "select"}
        onMouseDown={(e) => {
          e.cancelBubble = true;
          setSelectedObjectId(inset.id);
        }}
        onDragEnd={(e) => {
          setInsetImages((current) =>
            current.map((existing) =>
              existing.id === inset.id
                ? { ...existing, x: e.target.x(), y: e.target.y() }
                : existing
            )
          );
        }}
      >
        {img && <KonvaImage image={img} width={inset.width} height={inset.height} />}

        <Rect
          width={inset.width}
          height={inset.height}
          stroke={HALO_COLOR}
          strokeWidth={s(8)}
          listening={false}
        />

        <Rect
          width={inset.width}
          height={inset.height}
          stroke={isSelected ? SELECTED_COLOR : ANNOTATION_COLOR}
          strokeWidth={s(isSelected ? 4 : 2)}
        />
      </Group>

      {isSelected && tool === "select" && (
        <>
          <Rect
            x={inset.x + inset.width - 8}
            y={inset.y + inset.height - 8}
            width={16}
            height={16}
            fill={SELECTED_COLOR}
            stroke="white"
            strokeWidth={2}
            cornerRadius={3}
            draggable
            onMouseDown={(e) => {
              e.cancelBubble = true;
              setSelectedObjectId(inset.id);
            }}
            onDragMove={(e) => {
              const right = e.target.x() + 8;
              const rawWidth = right - inset.x;

              const newWidth = Math.max(80, rawWidth);
              const newHeight = newWidth / inset.aspectRatio;

              setInsetImages((current) =>
                current.map((existing) =>
                  existing.id === inset.id
                    ? { ...existing, width: newWidth, height: newHeight }
                    : existing
                )
              );
            }}
          />

          <Circle
            x={inset.leaderAnchorX}
            y={inset.leaderAnchorY}
            radius={s(7)}
            fill={SELECTED_COLOR}
            stroke="white"
            strokeWidth={2}
            draggable
            onMouseDown={(e) => {
              e.cancelBubble = true;
              setSelectedObjectId(inset.id);
            }}
            onDragMove={(e) => {
              const rawX = e.target.x();
              const rawY = e.target.y();

              setInsetImages((current) =>
                current.map((existing) => {
                  if (existing.id !== inset.id) return existing;

                  const centerX = existing.x + existing.width / 2;
                  const centerY = existing.y + existing.height / 2;

                  const snapped = shiftDown
                    ? snapEndpoint(centerX, centerY, rawX, rawY)
                    : { x: rawX, y: rawY };

                  return {
                    ...existing,
                    leaderAnchorX: snapped.x,
                    leaderAnchorY: snapped.y,
                  };
                })
              );
            }}
          />
        </>
      )}
    </React.Fragment>
  );
}

function polygonToPoints(polygon: { x: number; y: number }[]): number[] {
  return polygon.flatMap((point) => [point.x, point.y]);
}

function getPolygonBounds(polygon: { x: number; y: number }[]) {
  const xs = polygon.map((point) => point.x);
  const ys = polygon.map((point) => point.y);

  const minX = Math.min(...xs);
  const minY = Math.min(...ys);
  const maxX = Math.max(...xs);
  const maxY = Math.max(...ys);

  return {
    x: minX,
    y: minY,
    width: maxX - minX,
    height: maxY - minY,
  };
}

function ToolButton({
  active = false,
  onClick,
  disabled = false,
  children,
}: {
  active?: boolean;
  onClick: () => void;
  disabled?: boolean;
  children: React.ReactNode;
}) {
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      style={{
        background: active ? SELECTED_COLOR : "#2a2d2c",
        color: active ? "#0f172a" : "#fafafa",
        border: "1px solid #3a403e",
        borderRadius: 6,
        padding: "6px 10px",
        cursor: disabled ? "not-allowed" : "pointer",
        opacity: disabled ? 0.45 : 1,
        fontWeight: 600,
        fontSize: 13,
      }}
    >
      {children}
    </button>
  );
}

function App({
  imageSrc,
  debug = true,
  initialState,
  focusObjectsFromStreamlit,
  aiSuggestions,
  pendingInsetAsset,
  exportRequestId,
}: HeliosEditorProps) {

  const s = (value: number) => value / scale;
  const [image] = useImage(imageSrc ?? sampleImage);
  const canvasContainerRef = useRef<HTMLDivElement>(null);

  const stageRef = useRef<any>(null);
  const lastExportRequestIdRef = useRef<string | null>(null);

  const hasHydratedInitialStateRef = useRef(false);
  const isHydratingInitialStateRef = useRef(false);
  const autoSyncTimeoutRef = useRef<number | null>(null);

  const [dimensions, setDimensions] = useState({ width: 800, height: 600 });

  const [scale, setScale] = useState(1);

  const [tool, setTool] = useState<Tool>("select");
  const [isDrawing, setIsDrawing] = useState(false);
  const [selectedObjectId, setSelectedObjectId] = useState<string | null>(null);
  const [shiftDown, setShiftDown] = useState(false);

  const [lines, setLines] = useState<EditorLine[]>([]);

  const [callouts, setCallouts] = useState<EditorCallout[]>([]);

  const [insets, setInsets] = useState<EditorInset[]>([]);

  const [insetImages, setInsetImages] = useState<EditorInsetImage[]>([]);

  const [focusObjects, setFocusObjects] = useState<FocusObject[]>([
    {
      id: "focus_1",
      name: "Focus Object 1",
      haloEnabled: false,
      polygon: [
        { x: 430, y: 180 },
        { x: 540, y: 175 },
        { x: 610, y: 245 },
        { x: 570, y: 340 },
        { x: 440, y: 330 },
        { x: 390, y: 250 },
      ],
    },
    {
      id: "focus_2",
      name: "Focus Object 2",
      haloEnabled: false,
      polygon: [
        { x: 170, y: 140 },
        { x: 250, y: 130 },
        { x: 315, y: 190 },
        { x: 290, y: 260 },
        { x: 190, y: 270 },
        { x: 145, y: 205 },
      ],
    },
  ]);
  const streamlitDebug = debug;
  const selectedCallout = callouts.find((c) => c.id === selectedObjectId);
  const selectedInset = insets.find((i) => i.id === selectedObjectId);

  const selectedFocusObject = focusObjects.find(
    (obj) => obj.id === selectedObjectId
  );

  const getNextLineName = () => `Line ${lines.length + 1}`;
  const getNextCalloutName = () => `Callout ${callouts.length + 1}`;
  const getNextInsetName = () => `Detail View ${insets.length + 1}`;

  const deleteSelectedObject = () => {
    if (!selectedObjectId) return;

    setLines((current) => current.filter((line) => line.id !== selectedObjectId));
    setCallouts((current) =>
      current.filter((callout) => callout.id !== selectedObjectId)
    );
    setInsets((current) => current.filter((inset) => inset.id !== selectedObjectId));
    setInsetImages((current) =>
      current.filter((insetImage) => insetImage.id !== selectedObjectId)
    );

    setFocusObjects((current) =>
      current.filter((focusObject) => focusObject.id !== selectedObjectId)
    );

    setSelectedObjectId(null);
  };

  const getScaledPointer = (stage: any) => {
    const pointer = stage?.getPointerPosition();
    if (!pointer) return null;

    return {
      x: pointer.x / scale,
      y: pointer.y / scale,
    };
  };

  const createDetailViewFromFocusObject = (focusObject: FocusObject) => {
        const bounds = getPolygonBounds(focusObject.polygon);
        const padding = 24;

        const sourceX = Math.max(0, bounds.x - padding);
        const sourceY = Math.max(0, bounds.y - padding);
        const sourceWidth = bounds.width + padding * 2;
        const sourceHeight = bounds.height + padding * 2;

        const aspectRatio = sourceWidth / sourceHeight;

        const newWidth = 240;
        const newHeight = newWidth / aspectRatio;

        const newDetailView: EditorInset = {
          id: `inset_${Date.now()}`,
          name: getNextInsetName(),
          x: sourceX + sourceWidth + 40,
          y: sourceY,
          width: newWidth,
          height: newHeight,
          sourceX,
          sourceY,
          sourceWidth,
          sourceHeight,
          showLeader: true,
          aspectRatio,
          leaderAnchorX: sourceX + sourceWidth / 2,
          leaderAnchorY: sourceY + sourceHeight / 2,
        };

        setInsets((prev) => [...prev, newDetailView]);
        setSelectedObjectId(newDetailView.id);
        setTool("select");
      };

  const getEditorState = (): HeliosEditorState => ({
    version: 1,
    lines,
    callouts,
    detailViews: insets,
    insetImages,
    focusObjects,
    selectedObjectId,
  });

  useEffect(() => {
    if (!exportRequestId) return;
    if (lastExportRequestIdRef.current === exportRequestId) return;
    if (!stageRef.current || !image || scale <= 0) return;

    lastExportRequestIdRef.current = exportRequestId;

    const previousSelection = selectedObjectId;
    setSelectedObjectId(null);

    requestAnimationFrame(() => {
      requestAnimationFrame(() => {
        const dataUrl = stageRef.current.toDataURL({
          pixelRatio: 1 / scale,
          mimeType: "image/png",
        });

        setSelectedObjectId(previousSelection);

        Streamlit.setComponentValue({
          ...getEditorState(),
          selectedObjectId: previousSelection,
          exportedImageDataUrl: dataUrl,
          exportRequestId,
        });
      });
    });
  }, [
    exportRequestId,
    image,
    scale,
    selectedObjectId,
    lines,
    callouts,
    insets,
    insetImages,
    focusObjects,
  ]);

  useEffect(() => {
    if (!hasHydratedInitialStateRef.current) return;
    if (isHydratingInitialStateRef.current) return;

    if (autoSyncTimeoutRef.current !== null) {
      window.clearTimeout(autoSyncTimeoutRef.current);
    }

    autoSyncTimeoutRef.current = window.setTimeout(() => {
      try {
        Streamlit.setComponentValue(getEditorState());
      } catch {
        // Running standalone in Vite.
      }
    }, 250);

    return () => {
      if (autoSyncTimeoutRef.current !== null) {
        window.clearTimeout(autoSyncTimeoutRef.current);
      }
    };
  }, [
    lines,
    callouts,
    insets,
    insetImages,
    focusObjects,
    selectedObjectId,
  ]);

  useEffect(() => {
    if (hasHydratedInitialStateRef.current) return;

    isHydratingInitialStateRef.current = true;

    if (initialState) {
      if (initialState.lines) setLines(initialState.lines);
      if (initialState.callouts) setCallouts(initialState.callouts);
      if (initialState.detailViews) setInsets(initialState.detailViews);
      if (initialState.insetImages) setInsetImages(initialState.insetImages);
      if (initialState.focusObjects) setFocusObjects(initialState.focusObjects);

      if ("selectedObjectId" in initialState) {
        setSelectedObjectId(initialState.selectedObjectId ?? null);
      }
    }

    hasHydratedInitialStateRef.current = true;

    requestAnimationFrame(() => {
      isHydratingInitialStateRef.current = false;
    });
  }, [initialState]);

  useEffect(() => {
    const updateScale = () => {
      if (!image || !canvasContainerRef.current) return;

      const containerWidth = canvasContainerRef.current.offsetWidth;
      if (containerWidth <= 0) return;

      setDimensions({
        width: image.width,
        height: image.height,
      });

      setScale(containerWidth / image.width);
    };

    updateScale();

    window.addEventListener("resize", updateScale);
    const timeout = window.setTimeout(updateScale, 100);

    return () => {
      window.removeEventListener("resize", updateScale);
      window.clearTimeout(timeout);
    };
  }, [image]);

  useEffect(() => {
    if (!focusObjectsFromStreamlit) return;

    setFocusObjects((current) =>
      focusObjectsFromStreamlit.map((incoming) => {
        const existing = current.find((obj) => obj.id === incoming.id);

        return {
          ...incoming,
          haloEnabled: existing?.haloEnabled ?? incoming.haloEnabled ?? false,
        };
      })
    );
  }, [focusObjectsFromStreamlit]);

  useEffect(() => {
    try {
      Streamlit.setFrameHeight(950);
    } catch {
      // Running standalone in Vite.
    }
  }, []);

  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      const tag = (event.target as HTMLElement)?.tagName;
      if (tag === "INPUT" || tag === "TEXTAREA") return;

      if (event.key === "Shift") setShiftDown(true);

      if (event.key === "Delete" || event.key === "Backspace") {
        deleteSelectedObject();
      }

      if (event.key.toLowerCase() === "s") setTool("select");
      if (event.key.toLowerCase() === "d") setTool("draw");
      if (event.key.toLowerCase() === "c") setTool("callout");
      if (event.key.toLowerCase() === "i") setTool("inset");
      if (event.key.toLowerCase() === "v") setTool("detail");
      if (event.key.toLowerCase() === "f") setTool("focus");
    };

    const handleKeyUp = (event: KeyboardEvent) => {
      if (event.key === "Shift") setShiftDown(false);
    };

    window.addEventListener("keydown", handleKeyDown);
    window.addEventListener("keyup", handleKeyUp);

    return () => {
      window.removeEventListener("keydown", handleKeyDown);
      window.removeEventListener("keyup", handleKeyUp);
    };
  }, [selectedObjectId, lines, callouts, insets]);

  return (
    <div
      style={{
        width: "100%",
        height: "100%",
        background: "#222222",
        color: "#fafafa",
        padding: 0,
        margin: 0,
        fontFamily: "Inter, system-ui, sans-serif",
        boxSizing: "border-box",
      }}
    >
      <div
        style={{
          display: "flex",
          gap: 20,
          marginBottom: 16,
          alignItems: "center",
        }}
      >
        <div>
          <div style={{ fontSize: 11, color: "#888", marginBottom: 4 }}>
            Selection
          </div>
          <div style={{ display: "flex", gap: 6 }}>
            <ToolButton active={tool === "select"} onClick={() => setTool("select")}>
              Select (S)
            </ToolButton>

            <ToolButton active={tool === "focus"} onClick={() => setTool("focus")}>
              Focus (F)
            </ToolButton>
          </div>
        </div>

        <div>
          <div style={{ fontSize: 11, color: "#888", marginBottom: 4 }}>
            Annotation
          </div>
          <div style={{ display: "flex", gap: 6 }}>
            <ToolButton active={tool === "draw"} onClick={() => setTool("draw")}>
              Line (D)
            </ToolButton>

            <ToolButton active={tool === "callout"} onClick={() => setTool("callout")}>
              Callout (C)
            </ToolButton>
          </div>
        </div>

        <div>
          <div style={{ fontSize: 11, color: "#888", marginBottom: 4 }}>
            Views
          </div>
          <div style={{ display: "flex", gap: 6 }}>
            <ToolButton active={tool === "detail"} onClick={() => setTool("detail")}>
              Detail (V)
            </ToolButton>

            <ToolButton active={tool === "inset"} onClick={() => setTool("inset")}>
              Inset (I)
            </ToolButton>
          </div>
        </div>

        <div>
          <div style={{ fontSize: 11, color: "#888", marginBottom: 4 }}>
            Actions
          </div>
          <div style={{ display: "flex", gap: 6 }}>

            <ToolButton
              onClick={deleteSelectedObject}
              disabled={!selectedObjectId}
            >
              Delete
            </ToolButton>
          </div>
        </div>
      </div>

      <div
        style={{
          display: "flex",
          gap: 24,
          alignItems: "flex-start",
          background: "#222222",
        }}
      >
        <div
          ref={canvasContainerRef}
          style={{
            flex: "1 1 0",
            minWidth: 0,
            maxWidth: "100%",
            overflow: "hidden",
          }}
        >
          <Stage
            ref={stageRef}
            width={dimensions.width * scale}
            height={dimensions.height * scale}
            scaleX={scale}
            scaleY={scale}
            onMouseDown={(e) => {
                const stage = e.target.getStage();
                const pointer = getScaledPointer(stage);

                if (!stage || !pointer) return;

                const clickedOnCanvas = true;

                if (tool === "callout" && clickedOnCanvas) {
                  const offset = 60;

                  const newCallout: EditorCallout = {
                    id: `callout_${Date.now()}`,
                    name: getNextCalloutName(),
                    label: String.fromCharCode(65 + callouts.length),
                    anchorX: pointer.x,
                    anchorY: pointer.y,
                    circleX: pointer.x + offset,
                    circleY: pointer.y - offset,
                  };

                  setCallouts((prev) => [...prev, newCallout]);
                  setSelectedObjectId(newCallout.id);
                  return;
                }

                if (tool === "detail" && clickedOnCanvas) {
                  const newDetailView: EditorInset = {
                    id: `inset_${Date.now()}`,
                    name: getNextInsetName(),
                    x: pointer.x + 40,
                    y: pointer.y - 80,
                    width: 220,
                    height: 150,
                    sourceX: Math.max(0, pointer.x - 130),
                    sourceY: Math.max(0, pointer.y - 90),
                    sourceWidth: 260,
                    sourceHeight: 180,
                    showLeader: true,
                    aspectRatio: 220 / 150,
                    leaderAnchorX: pointer.x,
                    leaderAnchorY: pointer.y,
                  };

                  setInsets((prev) => [...prev, newDetailView]);
                  setSelectedObjectId(newDetailView.id);
                  return;
                }

                if (tool === "inset" && clickedOnCanvas) {
                  if (!pendingInsetAsset) {
                    alert("Process an inset image in Streamlit first.");
                    return;
                  }

                  const newInsetImage: EditorInsetImage = {
                    id: `inset_img_${Date.now()}`,
                    name: pendingInsetAsset.name,
                    x: pointer.x + 40,
                    y: pointer.y - 80,
                    width: 220,
                    height: 150,
                    imageSrc: pendingInsetAsset.imageSrc,
                    aspectRatio: 220 / 150,
                    showLeader: true,
                    leaderAnchorX: pointer.x,
                    leaderAnchorY: pointer.y,
                  };

                  setInsetImages((prev) => [...prev, newInsetImage]);
                  setSelectedObjectId(newInsetImage.id);
                  setTool("select");
                  return;
                }

                if (tool === "draw" && clickedOnCanvas) {
                  const newLine: EditorLine = {
                    id: `line_${Date.now()}`,
                    name: getNextLineName(),
                    x1: pointer.x,
                    y1: pointer.y,
                    x2: pointer.x,
                    y2: pointer.y,
                  };

                  setLines((prev) => [...prev, newLine]);
                  setSelectedObjectId(newLine.id);
                  setIsDrawing(true);
                  return;
                }

                setSelectedObjectId(null);
              }}
              onMouseMove={(e) => {
                if (!isDrawing) return;

                const stage = e.target.getStage();
                const pointer = getScaledPointer(stage);
                if (!pointer) return;

                setLines((prev) => {
                  const updated = [...prev];
                  const lastIndex = updated.length - 1;
                  const lastLine = updated[lastIndex];

                  const snapped = shiftDown
                    ? snapEndpoint(lastLine.x1, lastLine.y1, pointer.x, pointer.y)
                    : { x: pointer.x, y: pointer.y };

                  updated[lastIndex] = {
                    ...lastLine,
                    x2: snapped.x,
                    y2: snapped.y,
                  };

                  return updated;
                });
              }}
              onMouseUp={() => setIsDrawing(false)}
            >
              <Layer>
                {image && (
                  <KonvaImage
                    image={image}
                    width={dimensions.width}
                    height={dimensions.height}
                  />
                )}

                {aiSuggestions?.movement_lines?.map((line, index) => (
                  <React.Fragment key={`ai-line-${index}`}>
                    <Line
                      points={[line.start[0], line.start[1], line.end[0], line.end[1]]}
                      stroke={HALO_COLOR}
                      strokeWidth={7}
                      lineCap="round"
                      lineJoin="round"
                      opacity={0.7}
                      listening={false}
                    />
                    <Line
                      points={[line.start[0], line.start[1], line.end[0], line.end[1]]}
                      stroke="#777"
                      strokeWidth={2}
                      dash={[s(8), s(6)]}
                      lineCap="round"
                      lineJoin="round"
                      opacity={0.9}
                      listening={false}
                    />
                  </React.Fragment>
                ))}

                {aiSuggestions?.callouts?.map((callout, index) => (
                  <React.Fragment key={`ai-callout-${index}`}>
                    <Line
                      points={[
                        callout.circle[0],
                        callout.circle[1],
                        callout.end[0],
                        callout.end[1],
                      ]}
                      stroke={HALO_COLOR}
                      strokeWidth={7}
                      lineCap="round"
                      lineJoin="round"
                      opacity={0.7}
                      listening={false}
                    />

                    <Line
                      points={[
                        callout.circle[0],
                        callout.circle[1],
                        callout.end[0],
                        callout.end[1],
                      ]}
                      stroke="#777"
                      strokeWidth={2}
                      dash={[s(8), s(6)]}
                      lineCap="round"
                      lineJoin="round"
                      opacity={0.9}
                      listening={false}
                    />

                    <Circle
                      x={callout.circle[0]}
                      y={callout.circle[1]}
                      radius={s(18)}
                      fill="white"
                      stroke="#777"
                      strokeWidth={2}
                      dash={[6, 4]}
                      opacity={0.9}
                      listening={false}
                    />

                    <Text
                      x={callout.circle[0] - 5}
                      y={callout.circle[1] - 8}
                      text={callout.label}
                      fontSize={18}
                      fontStyle="bold"
                      fill="#777"
                      listening={false}
                    />
                  </React.Fragment>
                ))}

                {focusObjects.map((focusObject) => {
                  const isSelected = selectedObjectId === focusObject.id;
                  const points = polygonToPoints(focusObject.polygon);
                  const showSelectableOutline = tool === "focus" || isSelected;

                  return (
                    <React.Fragment key={focusObject.id}>
                      {focusObject.haloEnabled && (
                        <>
                          <Line
                            points={points}
                            closed
                            stroke={HALO_COLOR}
                            strokeWidth={3} // slightly thinner
                            lineJoin="round"
                            lineCap="round"
                            opacity={0.95}
                            listening={false}
                          />
                          <Line
                            points={points}
                            closed
                            stroke={HALO_COLOR}
                            strokeWidth={2}
                            lineJoin="round"
                            lineCap="round"
                            opacity={0.35}        // softer
                            shadowColor={HALO_COLOR}
                            shadowBlur={10}      // stronger outward glow
                            shadowOffsetX={0}
                            shadowOffsetY={0}
                            listening={false}
                          />
                        </>
                      )}

                      {showSelectableOutline && (
                        <Line
                          points={points}
                          closed
                          stroke={isSelected ? SELECTED_COLOR : "#ffffff"}
                          strokeWidth={s(isSelected ? 4 : 2)}
                          dash={isSelected ? undefined : [8, 6]}
                          lineJoin="round"
                          lineCap="round"
                          opacity={isSelected ? 1 : 0.7}
                          fill="rgba(80, 174, 145, 0.08)"
                          hitStrokeWidth={20}
                          onMouseDown={(e) => {
                            e.cancelBubble = true;
                            setSelectedObjectId(focusObject.id);
                            setTool("focus");
                          }}
                        />
                      )}

                    </React.Fragment>
                  );
                })}


                {insets.map((inset) => {
                  const isSelected = selectedObjectId === inset.id;

                  const insetCenterX = inset.x + inset.width / 2;
                  const insetCenterY = inset.y + inset.height / 2;

                  return (
                    <React.Fragment key={`source-${inset.id}`}>
                      {inset.showLeader && (
                        <>
                          <Line
                            points={[inset.leaderAnchorX, inset.leaderAnchorY, insetCenterX, insetCenterY]}
                            stroke={HALO_COLOR}
                            strokeWidth={s(isSelected ? 10 : 8)}
                            lineCap="round"
                            lineJoin="round"
                            listening={false}
                          />

                          <Line
                            points={[inset.leaderAnchorX, inset.leaderAnchorY, insetCenterX, insetCenterY]}
                            stroke={isSelected ? SELECTED_COLOR : ANNOTATION_COLOR}
                            strokeWidth={s(isSelected ? 4 : 3)}
                            lineCap="round"
                            lineJoin="round"
                            listening={false}
                          />
                        </>
                      )}

                      {isSelected && (
                        <>
                          <Rect
                            x={inset.sourceX}
                            y={inset.sourceY}
                            width={inset.sourceWidth}
                            height={inset.sourceHeight}
                            stroke={HALO_COLOR}
                            strokeWidth={s(8)}
                            dash={[10, 6]}
                            listening={false}
                          />
                          <Rect
                            x={inset.sourceX}
                            y={inset.sourceY}
                            width={inset.sourceWidth}
                            height={inset.sourceHeight}
                            stroke={SELECTED_COLOR}
                            strokeWidth={2}
                            dash={[10, 6]}
                            listening={false}
                          />
                        </>
                      )}
                      {isSelected && tool === "select" && (
                        <Circle
                          x={inset.leaderAnchorX}
                          y={inset.leaderAnchorY}
                          radius={s(7)}
                          fill={SELECTED_COLOR}
                          stroke="white"
                          strokeWidth={2}
                          draggable
                          onMouseDown={(e) => {
                            e.cancelBubble = true;
                            setSelectedObjectId(inset.id);
                          }}
                          onDragMove={(e) => {
                            const rawX = e.target.x();
                            const rawY = e.target.y();

                            setInsets((current) =>
                              current.map((existingInset) => {
                                if (existingInset.id !== inset.id) return existingInset;

                                const insetCenterX = existingInset.x + existingInset.width / 2;
                                const insetCenterY = existingInset.y + existingInset.height / 2;

                                const snapped = shiftDown
                                  ? snapEndpoint(insetCenterX, insetCenterY, rawX, rawY)
                                  : { x: rawX, y: rawY };

                                return {
                                  ...existingInset,
                                  leaderAnchorX: snapped.x,
                                  leaderAnchorY: snapped.y,
                                };
                              })
                            );
                          }}
                        />
                      )}
                    </React.Fragment>
                  );
                })}


                {insetImages.map((inset) => (
                  <InsetImageObject
                    key={inset.id}
                    inset={inset}
                    isSelected={selectedObjectId === inset.id}
                    tool={tool}
                    shiftDown={shiftDown}
                    scale={scale}
                    setSelectedObjectId={setSelectedObjectId}
                    setInsetImages={setInsetImages}
                  />
                ))}

                {lines.map((line) => {
                  const isSelected = selectedObjectId === line.id;

                  return (
                    <React.Fragment key={line.id}>
                      <Line
                        points={[line.x1, line.y1, line.x2, line.y2]}
                        stroke={HALO_COLOR}
                        strokeWidth={s(isSelected ? 10 : 8)}
                        lineCap="round"
                        lineJoin="round"
                        listening={false}
                      />

                      <Line
                        points={[line.x1, line.y1, line.x2, line.y2]}
                        stroke={isSelected ? SELECTED_COLOR : ANNOTATION_COLOR}
                        strokeWidth={s(isSelected ? 4 : 3)}
                        lineCap="round"
                        lineJoin="round"
                        hitStrokeWidth={16}
                        draggable={tool === "select"}
                        onMouseDown={(e) => {
                          e.cancelBubble = true;
                          setSelectedObjectId(line.id);
                        }}
                        onDragEnd={(e) => {
                          const dx = e.target.x();
                          const dy = e.target.y();

                          setLines((current) =>
                            current.map((existingLine) =>
                              existingLine.id === line.id
                                ? {
                                    ...existingLine,
                                    x1: existingLine.x1 + dx,
                                    y1: existingLine.y1 + dy,
                                    x2: existingLine.x2 + dx,
                                    y2: existingLine.y2 + dy,
                                  }
                                : existingLine
                            )
                          );

                          e.target.position({ x: 0, y: 0 });
                        }}
                      />

                      {isSelected && tool === "select" && (
                        <>
                          {[{ x: line.x1, y: line.y1, point: "start" }, { x: line.x2, y: line.y2, point: "end" }].map(
                            (handle) => (
                              <Circle
                                key={handle.point}
                                x={handle.x}
                                y={handle.y}
                                radius={s(7)}
                                fill={SELECTED_COLOR}
                                stroke="white"
                                strokeWidth={2}
                                draggable
                                onMouseDown={(e) => {
                                  e.cancelBubble = true;
                                  setSelectedObjectId(line.id);
                                }}
                                onDragMove={(e) => {
                                  const rawX = e.target.x();
                                  const rawY = e.target.y();

                                  setLines((current) =>
                                    current.map((existingLine) => {
                                      if (existingLine.id !== line.id) return existingLine;

                                      const fixedX =
                                        handle.point === "start"
                                          ? existingLine.x2
                                          : existingLine.x1;
                                      const fixedY =
                                        handle.point === "start"
                                          ? existingLine.y2
                                          : existingLine.y1;

                                      const snapped = shiftDown
                                        ? snapEndpoint(fixedX, fixedY, rawX, rawY)
                                        : { x: rawX, y: rawY };

                                      return handle.point === "start"
                                        ? {
                                            ...existingLine,
                                            x1: snapped.x,
                                            y1: snapped.y,
                                          }
                                        : {
                                            ...existingLine,
                                            x2: snapped.x,
                                            y2: snapped.y,
                                          };
                                    })
                                  );
                                }}
                              />
                            )
                          )}
                        </>
                      )}
                    </React.Fragment>
                  );
                })}

                {callouts.map((callout) => {
                  const isSelected = selectedObjectId === callout.id;

                  return (
                    <React.Fragment key={callout.id}>
                      <Line
                        points={[
                          callout.circleX,
                          callout.circleY,
                          callout.anchorX,
                          callout.anchorY,
                        ]}
                        stroke={HALO_COLOR}
                        strokeWidth={s(8)}
                        lineCap="round"
                        lineJoin="round"
                        listening={false}
                      />

                      <Line
                        points={[
                          callout.circleX,
                          callout.circleY,
                          callout.anchorX,
                          callout.anchorY,
                        ]}
                        stroke={isSelected ? SELECTED_COLOR : ANNOTATION_COLOR}
                        strokeWidth={s(3)}
                        hitStrokeWidth={s(16)}
                        onMouseDown={(e) => {
                          e.cancelBubble = true;
                          setSelectedObjectId(callout.id);
                        }}
                      />

                      <Circle
                        x={callout.circleX}
                        y={callout.circleY}
                        radius={s(22)}
                        fill={HALO_COLOR}
                        listening={false}
                      />

                      <Circle
                        x={callout.circleX}
                        y={callout.circleY}
                        radius={s(18)}
                        fill="white"
                        stroke={isSelected ? SELECTED_COLOR : ANNOTATION_COLOR}
                        strokeWidth={s(isSelected ? 3 : 2)}
                        draggable={tool === "select"}
                        onMouseDown={(e) => {
                          e.cancelBubble = true;
                          setSelectedObjectId(callout.id);
                        }}
                        onDragMove={(e) => {
                          setCallouts((current) =>
                            current.map((existingCallout) =>
                              existingCallout.id === callout.id
                                ? {
                                    ...existingCallout,
                                    circleX: e.target.x(),
                                    circleY: e.target.y(),
                                  }
                                : existingCallout
                            )
                          );
                        }}
                      />

                      <Text
                        x={callout.circleX}
                        y={callout.circleY}
                        text={callout.label}
                        fontSize={s(18)}
                        fontStyle="bold"
                        fill={ANNOTATION_COLOR}
                        listening={false}
                        width={s(36)}
                        height={s(36)}
                        align="center"
                        verticalAlign="middle"
                        offsetX={s(18)}
                        offsetY={s(18)}
                      />

                      {isSelected && tool === "select" && (
                        <Circle
                          x={callout.anchorX}
                          y={callout.anchorY}
                          radius={s(7)}
                          fill={SELECTED_COLOR}
                          stroke="white"
                          strokeWidth={s(2)}
                          draggable
                          onMouseDown={(e) => {
                            e.cancelBubble = true;
                            setSelectedObjectId(callout.id);
                          }}
                          onDragMove={(e) => {
                            setCallouts((current) =>
                              current.map((existingCallout) =>
                                existingCallout.id === callout.id
                                  ? {
                                      ...existingCallout,
                                      anchorX: e.target.x(),
                                      anchorY: e.target.y(),
                                    }
                                  : existingCallout
                              )
                            );
                          }}
                        />
                      )}
                    </React.Fragment>
                  );
                })}

                {insets.map((inset) => {
                  const isSelected = selectedObjectId === inset.id;

                  return (
                    <React.Fragment key={inset.id}>
                      <Group
                        x={inset.x}
                        y={inset.y}
                        draggable={tool === "select"}
                        onMouseDown={(e) => {
                          e.cancelBubble = true;
                          setSelectedObjectId(inset.id);
                        }}
                        onDragEnd={(e) => {
                          setInsets((current) =>
                            current.map((existingInset) =>
                              existingInset.id === inset.id
                                ? {
                                    ...existingInset,
                                    x: e.target.x(),
                                    y: e.target.y(),
                                  }
                                : existingInset
                            )
                          );
                        }}
                      >
                        {image && (
                          <KonvaImage
                            image={image}
                            width={inset.width}
                            height={inset.height}
                            crop={{
                              x: inset.sourceX,
                              y: inset.sourceY,
                              width: inset.sourceWidth,
                              height: inset.sourceHeight,
                            }}
                          />
                        )}

                        <Rect
                          width={inset.width}
                          height={inset.height}
                          stroke={HALO_COLOR}
                          strokeWidth={s(8)}
                          listening={false}
                        />

                        <Rect
                          width={inset.width}
                          height={inset.height}
                          stroke={isSelected ? SELECTED_COLOR : ANNOTATION_COLOR}
                          strokeWidth={s(isSelected ? 4 : 2)}
                        />

                      </Group>

                      {isSelected && tool === "select" && (
                        <Rect
                          x={inset.x + inset.width - 8}
                          y={inset.y + inset.height - 8}
                          width={16}
                          height={16}
                          fill={SELECTED_COLOR}
                          stroke="white"
                          strokeWidth={2}
                          cornerRadius={3}
                          draggable
                          onMouseDown={(e) => {
                            e.cancelBubble = true;
                            setSelectedObjectId(inset.id);
                          }}
                          onDragMove={(e) => {
                            const right = e.target.x() + 8;
                            const rawWidth = right - inset.x;

                            const newWidth = Math.max(80, rawWidth);
                            const newHeight = Math.max(60, newWidth / inset.aspectRatio);

                            setInsets((current) =>
                              current.map((existingInset) =>
                                existingInset.id === inset.id
                                  ? {
                                      ...existingInset,
                                      width: newWidth,
                                      height: newHeight,
                                    }
                                  : existingInset
                              )
                            );
                          }}
                        />
                      )}
                    </React.Fragment>
                  );
                })}
              </Layer>
          </Stage>
        </div>
        <div
          style={{
            width: 280,
            background: "#1c1d1d",
            border: "none",
            borderRadius: 12,
            padding: 16,
          }}
        >
          <h3 style={{ marginTop: 0 }}>Objects</h3>

          <h4>Selected</h4>

          {selectedCallout && (
                  <div style={{ marginBottom: 16, display: "flex", gap: 8 }}>
                    <label>Label:</label>
                    <input
                      value={selectedCallout.label}
                      onChange={(e) => {
                        const newLabel = e.target.value.slice(0, 3);

                        setCallouts((current) =>
                          current.map((callout) =>
                            callout.id === selectedCallout.id
                              ? { ...callout, label: newLabel }
                              : callout
                          )
                        );
                      }}
                      style={{
                        width: 64,
                        padding: "6px 8px",
                        borderRadius: 8,
                        border: "1px solid #3a403e",
                      }}
                    />
                  </div>
                )}

          {selectedInset && (
                  <div
                    style={{
                      marginBottom: 16,
                      display: "flex",
                      gap: 16,
                      alignItems: "center",
                    }}
                  >
                    <label>
                      <input
                        type="checkbox"
                        checked={selectedInset.showLeader}
                        onChange={(e) => {
                          setInsets((current) =>
                            current.map((inset) =>
                              inset.id === selectedInset.id
                                ? { ...inset, showLeader: e.target.checked }
                                : inset
                            )
                          );
                        }}
                      />{" "}
                      Show leader
                    </label>

                  </div>
                )}

          {selectedFocusObject && (
                  <button
                    onClick={() => createDetailViewFromFocusObject(selectedFocusObject)}
                    style={{
                      borderRadius: 8,
                      padding: "6px 10px",
                      cursor: "pointer",
                      fontWeight: 600,
                    }}
                  >
                    Create detail view from focus
                  </button>
                )}

          {focusObjects.find((obj) => obj.id === selectedObjectId) && (
                              <div
                                style={{
                                  marginBottom: 16,
                                  display: "flex",
                                  gap: 16,
                                  alignItems: "center",
                                }}
                              >
                                <label>
                                  <input
                                    type="checkbox"
                                    checked={
                                      focusObjects.find((obj) => obj.id === selectedObjectId)
                                        ?.haloEnabled ?? false
                                    }
                                    onChange={(e) => {
                                      const checked = e.target.checked;

                                      setFocusObjects((current) =>
                                        current.map((obj) =>
                                          obj.id === selectedObjectId
                                            ? { ...obj, haloEnabled: checked }
                                            : obj
                                        )
                                      );
                                    }}
                                  />{" "}
                                  Enable object halo
                                </label>
                              </div>
                            )}

          <h4>Lines</h4>
          {lines.map((line) => (
            <button
              key={line.id}
              onClick={() => setSelectedObjectId(line.id)}
              style={{
                width: "100%",
                marginBottom: 6,
                padding: "8px 10px",
                borderRadius: 8,
                border:
                  selectedObjectId === line.id
                    ? `1px solid ${SELECTED_COLOR}`
                    : "1px solid #343837",
                background: selectedObjectId === line.id ? SELECTED_COLOR : "#242827",
                color: selectedObjectId === line.id ? "#0f172a" : "#fafafa",
                textAlign: "left",
              }}
            >
              {line.name}
            </button>
          ))}

          <h4>Callouts</h4>
          {callouts.map((callout) => (
            <button
              key={callout.id}
              onClick={() => setSelectedObjectId(callout.id)}
              style={{
                width: "100%",
                marginBottom: 6,
                padding: "8px 10px",
                borderRadius: 8,
                border:
                  selectedObjectId === callout.id
                    ? `1px solid ${SELECTED_COLOR}`
                    : "1px solid #343837",
                background:
                  selectedObjectId === callout.id ? SELECTED_COLOR : "#242827",
                color: selectedObjectId === callout.id ? "#0f172a" : "#fafafa",
                textAlign: "left",
              }}
            >
              {callout.name} · {callout.label}
            </button>
          ))}

          <h4>Detail Views</h4>
          {insets.map((inset) => (
            <button
              key={inset.id}
              onClick={() => setSelectedObjectId(inset.id)}
              style={{
                width: "100%",
                marginBottom: 6,
                padding: "8px 10px",
                borderRadius: 8,
                border:
                  selectedObjectId === inset.id
                    ? `1px solid ${SELECTED_COLOR}`
                    : "1px solid #343837",
                background: selectedObjectId === inset.id ? SELECTED_COLOR : "#242827",
                color: selectedObjectId === inset.id ? "#0f172a" : "#fafafa",
                textAlign: "left",
              }}
            >
              {inset.name}
            </button>
          ))}

          <h4>Inset Images</h4>
          {insetImages.map((insetImage) => (
            <button
              key={insetImage.id}
              onClick={() => setSelectedObjectId(insetImage.id)}
              style={{
                width: "100%",
                marginBottom: 6,
                padding: "8px 10px",
                borderRadius: 8,
                border:
                  selectedObjectId === insetImage.id
                    ? `1px solid ${SELECTED_COLOR}`
                    : "1px solid #343837",
                background:
                  selectedObjectId === insetImage.id ? SELECTED_COLOR : "#242827",
                color: selectedObjectId === insetImage.id ? "#0f172a" : "#fafafa",
                textAlign: "left",
              }}
            >
              {insetImage.name}
            </button>
          ))}

          <h4>Focus Objects</h4>
          {focusObjects.map((focusObject) => (
            <button
              key={focusObject.id}
              onClick={() => {
                setSelectedObjectId(focusObject.id);
                setTool("focus");
              }}
              style={{
                width: "100%",
                marginBottom: 6,
                padding: "8px 10px",
                borderRadius: 8,
                border:
                  selectedObjectId === focusObject.id
                    ? `1px solid ${SELECTED_COLOR}`
                    : "1px solid #343837",
                background:
                  selectedObjectId === focusObject.id ? SELECTED_COLOR : "#242827",
                color: selectedObjectId === focusObject.id ? "#0f172a" : "#fafafa",
                textAlign: "left",
              }}
            >
              {focusObject.name}
              {focusObject.haloEnabled ? " · Halo" : ""}
            </button>
          ))}

          {streamlitDebug && (
            <>
              <h4>State</h4>
              <pre
                style={{
                  background: "#111",
                  padding: 12,
                  borderRadius: 8,
                  fontSize: 11,
                  maxHeight: 260,
                  overflow: "auto",
                }}
              >
                {JSON.stringify(getEditorState(), null, 2)}
              </pre>
            </>
          )}
        </div>
      </div>
    </div>
  );
}

function StreamlitAppWrapper() {
  const [args, setArgs] = useState<any>({});

  useEffect(() => {
    const onRender = (event: Event) => {
      const customEvent = event as CustomEvent;
      setArgs(customEvent.detail.args ?? {});
      Streamlit.setFrameHeight(950);
    };

    Streamlit.events.addEventListener(Streamlit.RENDER_EVENT, onRender);
    Streamlit.setComponentReady();
    Streamlit.setFrameHeight(950);

    return () => {
      Streamlit.events.removeEventListener(Streamlit.RENDER_EVENT, onRender);
    };
  }, []);

  return (
    <App
      imageSrc={args.imageSrc}
      debug={args.debug ?? false}
      initialState={args.initialState}
      focusObjectsFromStreamlit={args.focusObjectsFromStreamlit}
      aiSuggestions={args.aiSuggestions}
      pendingInsetAsset={args.pendingInsetAsset}
      exportRequestId={args.exportRequestId}
    />
  );
}

export default StreamlitAppWrapper;