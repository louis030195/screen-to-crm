# screen-activity-callback

classify whats happening on ur screen w computer vision & trigger ur callback

## usage

```python


def on_activity(activity: str):
  push_to_db(activity, {"time": time.time()})

sac(on_activity)
```

correlate ur activity w ur biotech wearable:


```python
def on_activity(activity: str):
  ouraring_tag(activity, {"time": time.time()}))

sac(on_activity)
```

```python
def on_activity(activity: str):
  brain_computer_interface_tag(activity, {"time": time.time()}))

sac(on_activity)
```


automation

```python
def on_activity(activity: str):
  bad = llm(f"is this activity bad for me? {activity}")
  if bad:
    initiate_autodestruction()

sac(on_activity)
```

```python
def on_activity(activity: str):
  good = llm(f"is this activity good for me? {activity}")
  if good:
    deliver_chocolate()

sac(on_activity)
```
