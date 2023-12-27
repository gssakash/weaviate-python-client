import os
from typing import Any, Optional, List, Generator, Protocol, Type, Dict, Tuple

import pytest

import weaviate
from weaviate import Collection
from weaviate.collections.classes.config import (
    Property,
    _VectorizerConfigCreate,
    _InvertedIndexConfigCreate,
    _ReferencePropertyBase,
    Configure,
    _GenerativeConfigCreate,
    _ReplicationConfigCreate,
    DataType,
    _MultiTenancyConfigCreate,
    _VectorIndexConfigCreate,
)
from weaviate.collections.classes.types import Properties


class CollectionFactory(Protocol):
    """Typing for fixture."""

    def __call__(
        self,
        name: str,
        properties: Optional[List[Property]] = None,
        references: Optional[List[_ReferencePropertyBase]] = None,
        vectorizer_config: Optional[_VectorizerConfigCreate] = None,
        inverted_index_config: Optional[_InvertedIndexConfigCreate] = None,
        multi_tenancy_config: Optional[_MultiTenancyConfigCreate] = None,
        generative_config: Optional[_GenerativeConfigCreate] = None,
        headers: Optional[Dict[str, str]] = None,
        ports: Tuple[int, int] = (8080, 50051),
        data_model_props: Optional[Type[Properties]] = None,
        data_model_refs: Optional[Type[Properties]] = None,
        replication_config: Optional[_ReplicationConfigCreate] = None,
        vector_index_config: Optional[_VectorIndexConfigCreate] = None,
        description: Optional[str] = None,
    ) -> Collection[Any, Any]:
        """Typing for fixture."""
        ...


@pytest.fixture
def collection_factory() -> Generator[CollectionFactory, None, None]:
    name_fixture: Optional[str] = None
    client_fixture: Optional[weaviate.WeaviateClient] = None

    def _factory(
        name: str,
        properties: Optional[List[Property]] = None,
        references: Optional[List[_ReferencePropertyBase]] = None,
        vectorizer_config: Optional[_VectorizerConfigCreate] = None,
        inverted_index_config: Optional[_InvertedIndexConfigCreate] = None,
        multi_tenancy_config: Optional[_MultiTenancyConfigCreate] = None,
        generative_config: Optional[_GenerativeConfigCreate] = None,
        headers: Optional[Dict[str, str]] = None,
        ports: Tuple[int, int] = (8080, 50051),
        data_model_props: Optional[Type[Properties]] = None,
        data_model_refs: Optional[Type[Properties]] = None,
        replication_config: Optional[_ReplicationConfigCreate] = None,
        vector_index_config: Optional[_VectorIndexConfigCreate] = None,
        description: Optional[str] = None,
    ) -> Collection[Any, Any]:
        nonlocal client_fixture, name_fixture
        name_fixture = _sanitize_collection_name(name)
        client_fixture = weaviate.connect_to_local(
            headers=headers, grpc_port=ports[1], port=ports[0]
        )
        client_fixture.collections.delete(name_fixture)

        collection: Collection[Any, Any] = client_fixture.collections.create(
            name=name_fixture,
            description=description,
            vectorizer_config=vectorizer_config,
            properties=properties,
            references=references,
            inverted_index_config=inverted_index_config,
            multi_tenancy_config=multi_tenancy_config,
            generative_config=generative_config,
            data_model_properties=data_model_props,
            data_model_references=data_model_refs,
            replication_config=replication_config,
            vector_index_config=vector_index_config,
        )
        return collection

    yield _factory
    if client_fixture is not None and name_fixture is not None:
        client_fixture.collections.delete(name_fixture)


class OpenAICollection(Protocol):
    """Typing for fixture."""

    def __call__(
        self, name: str, vectorizer_config: Optional[_VectorizerConfigCreate] = None
    ) -> Collection[Any, Any]:
        """Typing for fixture."""
        ...


@pytest.fixture
def openai_collection(
    collection_factory: CollectionFactory,
) -> Generator[OpenAICollection, None, None]:
    def _factory(
        name: str, vectorizer_config: Optional[_VectorizerConfigCreate] = None
    ) -> Collection[Any, Any]:
        api_key = os.environ.get("OPENAI_APIKEY")
        if api_key is None:
            pytest.skip("No OpenAI API key found.")

        if vectorizer_config is None:
            vectorizer_config = Configure.Vectorizer.none()

        collection = collection_factory(
            name=name,
            vectorizer_config=vectorizer_config,
            properties=[
                Property(name="text", data_type=DataType.TEXT),
                Property(name="content", data_type=DataType.TEXT),
                Property(name="extra", data_type=DataType.TEXT),
            ],
            generative_config=Configure.Generative.openai(),
            ports=(8086, 50057),
            headers={"X-OpenAI-Api-Key": api_key},
        )

        return collection

    yield _factory


class CollectionFactoryGet(Protocol):
    """Typing for fixture."""

    def __call__(
        self,
        name: str,
        data_model_props: Optional[Type[Properties]] = None,
        data_model_refs: Optional[Type[Properties]] = None,
    ) -> Collection[Any, Any]:
        """Typing for fixture."""
        ...


@pytest.fixture
def collection_factory_get() -> Generator[CollectionFactoryGet, None, None]:
    def _factory(
        name: str,
        data_model_props: Optional[Type[Properties]] = None,
        data_model_refs: Optional[Type[Properties]] = None,
    ) -> Collection[Any, Any]:
        client_fixture = weaviate.connect_to_local()

        collection: Collection[Any, Any] = client_fixture.collections.get(
            name=_sanitize_collection_name(name),
            data_model_properties=data_model_props,
            data_model_references=data_model_refs,
        )
        return collection

    yield _factory


def _sanitize_collection_name(name: str) -> str:
    name = name.replace("[", "").replace("]", "").replace("-", "").replace(" ", "")
    return name[0].upper() + name[1:]
