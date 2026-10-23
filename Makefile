SHELL := /bin/bash

WS := $(shell pwd)
ROS_DISTRO ?= jazzy
ROS_SETUP := /opt/ros/$(ROS_DISTRO)/setup.bash
LOCAL_SETUP := $(WS)/install/setup.bash
UNITY_BIN := $(WS)/src/simulation/unity_sim/Build_Ubuntu/AD_Sim.x86_64
UNITY_CONFIG := $(WS)/src/simulation/unity_sim/Build_Ubuntu/AD_Sim_Data/StreamingAssets/simulation_config.json

CAR_MODEL := $(WS)/src/ad_perception/trafficlight/output/runs/detect/car/weights/best.pt
TL_MODEL := $(WS)/src/ad_perception/trafficlight/output/runs/detect/trafficlight_box_yolov8s/weights/best.pt

PKGS := ad_interfaces simulation ad_perception ad_bringup ad_planning ad_decision dummy_controller
PERCEPTION_PKGS := ad_interfaces ad_perception ad_bringup ad_decision ad_planning
NUMERIC_GOALS := $(shell seq 0 200 2>/dev/null)
SPAWN_INDEX ?= $(firstword $(filter $(NUMERIC_GOALS),$(MAKECMDGOALS)))
ifeq ($(SPAWN_INDEX),)
SPAWN_INDEX := 0
endif

TL_RANGES ?= 1-5,70-77,87-101,126-131,145-150
CAR_RANGES ?= 39-63,101-105,149-150
ALL_RANGES ?= 1-5,33-39,39-63,70-77,87-101,101-105,126-131,145-150
BAG_TOPICS := /OurCar/Sensors/RGBCameraLeft/image_raw /OurCar/Sensors/DepthCamera/image_raw /OurCar/Sensors/SemanticCamera/image_raw /planning/waypoint_index /perception/traffic_light_state /perception/traffic_light_label /perception/traffic_light_detections /perception/front_car_distance /perception/debug/detection_combined

define ROS_ENV
source $(ROS_SETUP) && source $(LOCAL_SETUP)
endef

.PHONY: help build build-perception clean check-models set-spawn kill-stale install-yolo \
	vision drive all debug rviz collect-images collect-tl-images collect-car-images record-bag record-color-bag \
	echo-topics hz topics train-car stop status

help:
	@echo ""
	@echo "Build"
	@echo "  make build              Build all packages"
	@echo "  make build-perception   Build perception-related packages"
	@echo "  make clean              Remove build/install/log"
	@echo ""
	@echo "Main run"
	@echo "  make vision [N]         Full autonomous driving + visual debug windows, spawnIndex=N"
	@echo "  make drive [N]          Full autonomous driving only, no extra debug windows, spawnIndex=N"
	@echo "  make all                Alias of make vision"
	@echo "  make debug              Open visual debug windows only (no RViz)"
	@echo "  make rviz               Open the RViz point cloud/octomap/local_map view (separate, heavier)"
	@echo "  make set-spawn [N]      Only update Unity spawnIndex in simulation_config.json"
	@echo ""
	@echo "Data modes"
	@echo "  make collect-images     DISABLED: image dataset collection is no longer part of the formal workflow"
	@echo "  make collect-tl-images  DISABLED: traffic-light training images are already collected"
	@echo "  make collect-car-images DISABLED: car model is already trained"
	@echo "  make record-bag         Full autonomous driving + rosbag recording"
	@echo "  make record-color-bag N Rosbag mode for traffic-light color analysis at spawnIndex=N"
	@echo ""
	@echo "Useful spawn points"
	@echo "  0   light0 / full lap start"
	@echo "  33  candidate/negative light area"
	@echo "  39  car following section start"
	@echo "  70  light1"
	@echo "  87  light2"
	@echo "  101 car braking scene"
	@echo "  126 light3"
	@echo "  145 light4"
	@echo "  149 obstacle / end area"
	@echo ""
	@echo "Checks"
	@echo "  make echo-topics        Echo key perception/decision topics"
	@echo "  make hz                 Quick topic-rate check"
	@echo "  make topics             List relevant topics"
	@echo "  make check-models       Verify traffic-light and car YOLO weights exist"
	@echo ""
	@echo "Tools"
	@echo "  make install-yolo       Install ultralytics into current Python"
	@echo "  make train-car          Train car YOLO model"
	@echo "  make stop               Stop ROS/Unity/debug/bag processes started by this Makefile"
	@echo "  make status             git status --short"
	@echo ""

build:
	source $(ROS_SETUP) && colcon build --packages-select $(PKGS)

build-perception:
	source $(ROS_SETUP) && colcon build --packages-select $(PERCEPTION_PKGS)

clean:
	rm -rf build install log

check-models:
	@test -f "$(TL_MODEL)" || (echo "Missing traffic-light model: $(TL_MODEL)" && exit 1)
	@test -f "$(CAR_MODEL)" || (echo "Missing car model: $(CAR_MODEL)" && exit 1)
	@echo "OK: $(TL_MODEL)"
	@echo "OK: $(CAR_MODEL)"

set-spawn:
	python3 -c "import json, pathlib; p=pathlib.Path('$(UNITY_CONFIG)'); data=json.loads(p.read_text()); data.setdefault('spawnIndexData', {})['spawnIndex']=int('$(SPAWN_INDEX)'); data.setdefault('controlConfig', {})['controllerOverride']=False; data['controlConfig']['controllerOnly']=False; p.write_text(json.dumps(data, indent=4) + '\n'); print('Unity spawnIndex set to {}, controllerOverride=false: {}'.format(data['spawnIndexData']['spawnIndex'], p))"

# Kills any stack left running from a previous vision/drive/record-bag
# run before starting a new one, so Unity doesn't end up with two windows and
# the ROS graph doesn't end up with duplicate node names.
kill-stale:
	-@pkill -f "AD_Sim.x86_64" 2>/dev/null
	-@pkill -f "ros2 launch ad_bringup autonomous_driving.launch.py" 2>/dev/null
	-@pkill -f "ros2 launch simulation simulation.launch.py" 2>/dev/null
	-@pkill -f "dummy_controller_node" 2>/dev/null
	@sleep 1

install-yolo:
	python3 -m pip install ultralytics

train-car:
	cd $(WS)/src/ad_perception/trafficlight/training && python3 train_car_detector_yolov8n.py

drive: check-models kill-stale set-spawn
	$(ROS_ENV) && ros2 launch ad_bringup autonomous_driving.launch.py \
		enable_decision:=true \
		enable_controller:=true \
		enable_car_detector:=true \
		initial_waypoint_index:=$(SPAWN_INDEX) \
		initialize_from_nearest_waypoint:=true

# Your default perception workflow:
# full autonomous driving plus visual debug. Terminals intentionally do not
# use "exec bash"; after "make stop" kills the process, the window closes.
vision: check-models kill-stale set-spawn
	gnome-terminal --title="AD-Vision-FullStack" -- bash -lc "cd '$(WS)' && $(ROS_ENV) && ros2 launch ad_bringup autonomous_driving.launch.py enable_decision:=true enable_controller:=true enable_car_detector:=true initial_waypoint_index:=$(SPAWN_INDEX) initialize_from_nearest_waypoint:=true"
	sleep 10
	$(MAKE) debug

all: vision

debug:
	gnome-terminal --title="AD-Debug-Combined" -- bash -lc "cd '$(WS)' && $(ROS_ENV) && ros2 run rqt_image_view rqt_image_view /perception/debug/detection_combined"
	gnome-terminal --title="AD-Echo-Perception" -- bash -lc "cd '$(WS)' && $(ROS_ENV) && ros2 topic echo /perception/traffic_light_label"
	gnome-terminal --title="AD-Echo-CarDistance" -- bash -lc "cd '$(WS)' && $(ROS_ENV) && ros2 topic echo /perception/front_car_distance"
	gnome-terminal --title="AD-Echo-Decision" -- bash -lc "cd '$(WS)' && $(ROS_ENV) && ros2 topic echo /decision/state"

# Standalone RViz map/point cloud/octomap view, separate from `make debug` so
# it's not always running - it's the heaviest of the debug windows (Unity + 2
# CPU YOLO detectors already saturate a 6-core machine on their own).
rviz:
	gnome-terminal --title="AD-Debug-Map" -- bash -lc "cd '$(WS)' && $(ROS_ENV) && ros2 launch ad_perception perception_rviz.launch.py"

# Image/screenshot-style dataset collection is disabled for the formal
# workflow. The trained traffic-light and car models are already available;
# keep these targets only so old notes fail clearly instead of launching the
# historical trafficlight_dataset_collector.
collect-images:
	@echo "DISABLED: image dataset collection is no longer part of the formal workflow."

collect-tl-images:
	@echo "DISABLED: traffic-light training images are already collected."

collect-car-images:
	@echo "DISABLED: car model is already trained; use make vision to run car detection."

# Rosbag recording mode. This is different from image collection: it records
# raw topics for later replay/offline extraction.
record-bag: check-models kill-stale set-spawn
	gnome-terminal --title="AD-Bag-FullStack" -- bash -lc "cd '$(WS)' && $(ROS_ENV) && ros2 launch ad_bringup autonomous_driving.launch.py enable_decision:=true enable_controller:=true enable_car_detector:=true initial_waypoint_index:=$(SPAWN_INDEX) initialize_from_nearest_waypoint:=true"
	sleep 10
	gnome-terminal --title="AD-Bag-Record" -- bash -lc "cd '$(WS)' && $(ROS_ENV) && mkdir -p src/ad_perception/rosbags && ros2 bag record -o src/ad_perception/rosbags/ad_run_$$(date +%Y%m%d_%H%M%S) $(BAG_TOPICS)"
	$(MAKE) debug

record-color-bag: check-models kill-stale set-spawn
	gnome-terminal --title="AD-ColorBag-FullStack" -- bash -lc "cd '$(WS)' && $(ROS_ENV) && ros2 launch ad_bringup autonomous_driving.launch.py enable_decision:=true enable_controller:=true enable_car_detector:=true initial_waypoint_index:=$(SPAWN_INDEX) initialize_from_nearest_waypoint:=true"
	sleep 10
	gnome-terminal --title="AD-ColorBag-Record" -- bash -lc "cd '$(WS)' && $(ROS_ENV) && mkdir -p src/ad_perception/datasets/traffic_light_color_eval/bags && ros2 bag record -o src/ad_perception/datasets/traffic_light_color_eval/bags/spawn$(SPAWN_INDEX)_$$(date +%Y%m%d_%H%M%S) $(BAG_TOPICS)"
	$(MAKE) debug

echo-topics:
	$(ROS_ENV) && \
		(ros2 topic echo /perception/traffic_light_label & \
		 ros2 topic echo /perception/front_car_distance & \
		 ros2 topic echo /decision/state; \
		 wait)

hz:
	$(ROS_ENV) && \
		echo "--- RGB left ---" && timeout 3 ros2 topic hz /OurCar/Sensors/RGBCameraLeft/image_raw || true; \
		echo "--- Depth ---" && timeout 3 ros2 topic hz /OurCar/Sensors/DepthCamera/image_raw || true; \
		echo "--- Traffic light state ---" && timeout 3 ros2 topic hz /perception/traffic_light_state || true; \
		echo "--- Front car distance ---" && timeout 3 ros2 topic hz /perception/front_car_distance || true; \
		echo "--- Combined debug ---" && timeout 3 ros2 topic hz /perception/debug/detection_combined || true

topics:
	$(ROS_ENV) && ros2 topic list | grep -E "OurCar|perception|planning|decision|cmd|control" || true

stop: kill-stale
	@echo "Stopping AD ROS/Unity/debug/bag processes..."
	-pkill -f "ros2 run rqt_image_view rqt_image_view"
	-pkill -f "rqt_image_view"
	-pkill -f "perception_debug.rviz"
	-pkill -f "ros2 topic echo /perception"
	-pkill -f "ros2 topic echo /decision"
	-pkill -INT -f "ros2 bag record"
	@sleep 2
	-pkill -f "ros2 bag record"
	@echo "Done. New Makefile terminals close automatically because they do not keep an idle shell open."

status:
	git status --short

$(NUMERIC_GOALS):
	@:
