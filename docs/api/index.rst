API Reference
=============

Data Apps
---------

Data apps are the primary execution interface. All concrete apps follow the
same context-manager pattern::

    with MyApp(config) as app:
        app.run()

Base
~~~~

.. automodule:: threephi_framework.data_apps.base
   :members:
   :undoc-members:
   :show-inheritance:

Timeseries Ingestor
~~~~~~~~~~~~~~~~~~~

.. automodule:: threephi_framework.data_apps.timeseries_ingestor
   :members:
   :undoc-members:
   :show-inheritance:

Topology Ingestor
~~~~~~~~~~~~~~~~~

.. automodule:: threephi_framework.data_apps.topology_ingestor
   :members:
   :undoc-members:
   :show-inheritance:

Topology Tester
~~~~~~~~~~~~~~~

.. automodule:: threephi_framework.data_apps.topology_tester
   :members:
   :undoc-members:
   :show-inheritance:

SM Classifier
~~~~~~~~~~~~~

.. automodule:: threephi_framework.data_apps.sm_classifier
   :members:
   :undoc-members:
   :show-inheritance:

Controllers
-----------

Controllers provide a programmatic API for querying and manipulating data,
using SQL for relational data and the relevant protocols (for example, S3) for file-based data.
They are used directly when integrating with orchestration tools such as
Airflow, or when building custom workflows on top of the framework.

Meta Controller
~~~~~~~~~~~~~~~

.. automodule:: threephi_framework.controllers.meta
   :members:
   :undoc-members:
   :show-inheritance:

Topology Controller
~~~~~~~~~~~~~~~~~~~

.. automodule:: threephi_framework.controllers.topology
   :members:
   :undoc-members:
   :show-inheritance:

Time Series Controller
~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: threephi_framework.controllers.time_series
   :members:
   :undoc-members:
   :show-inheritance:

Object Storage
--------------

Storage connectors abstract the underlying object storage backend. The
:class:`~threephi_framework.object_storage.base_connector.BaseConnector`
interface defines the full contract; swap the concrete implementation to
change the storage backend without touching application code.

Base Connector
~~~~~~~~~~~~~~

.. automodule:: threephi_framework.object_storage.base_connector
   :members:
   :undoc-members:
   :show-inheritance:

S3 Connector
~~~~~~~~~~~~

.. automodule:: threephi_framework.object_storage.s3_connector
   :members:
   :undoc-members:
   :show-inheritance:

Azure Blob Connector
~~~~~~~~~~~~~~~~~~~~

.. automodule:: threephi_framework.object_storage.azure_blob_connector
   :members:
   :undoc-members:
   :show-inheritance:
